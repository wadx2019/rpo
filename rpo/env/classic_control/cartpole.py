"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Modified for swing up problem with constraints
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import torch
import warnings

class CartSafeEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
        The pendulum starts hanging down, and the goal is to swing it up by increasing and reducing the cart's velocity.

    Source:
        This environment is a modified version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 0              2*math.pi
        3	Pole Velocity At Tip      -Inf            Inf

        Note: angle state is only up to 2*Pi, meaning e.g. 1*Pi is the same state as 3*Pi

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Since we want the agent to learn to swing up our Reward is 1+cos(Pole Angle) for every step taken,
        including the termination step

    Constraint Cost:
        We constrain the position of the cart. Every time step that the cart violates these constraints we return an
        immediate cost of 1 for the violated constraint. For Constrained Markov Decision Problems (CMDPs) we typically
        want to have a threshold for the expected cumulative constraint costs.
        The immediate constraint costs are returned inside the info dict.
        Constraints:
            -1 < Cart Position < 1


    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05], for the angle we add pi such that the pendulum
        starts at the bottom and the environment is consequently more difficult

    Episode Termination:
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 300
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 520 over 100 consecutive trials.
            A reward of 520 corresponds approx to a lower bound for an epsiode of 300 steps with 75% of the time in between an angle of [-12,12] degrees.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.mu_c = 0.1
        self.mu_p = 0.01
        self.delta = np.array([np.pi/3, -np.pi/6])  # force angle (-\pi, \pi)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.x_constraint = 1.0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(np.array([-10, -10]).astype(np.float32), np.array([10, 10]).astype(np.float32))

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.eq_num = 1
        self.ineq_num = 6

        self.seed()
        self.viewer = None
        self.state = None

        self.partial_actions = np.random.choice(self.action_dim, self.action_dim-self.eq_num, replace=False)
        self.other_actions = np.setdiff1d(np.arange(self.action_dim), self.partial_actions)

        self.steps_beyond_done = None

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.diff_eq = torch.sin(torch.tensor(self.delta, dtype=torch.float32, device=self.device)).view(self.eq_num, self.action_dim)
        self.diff_eq_partial = self.diff_eq[:, self.partial_actions]
        self.diff_eq_other_inv = torch.inverse(self.diff_eq[:, self.other_actions])
        self.diff_eq_bias = torch.tensor([0], dtype=torch.float32, device=self.device).view(self.eq_num, 1)

        self.diff_ineq = torch.cos(torch.tensor(self.delta, dtype=torch.float32, device=self.device)).repeat(self.ineq_num-2*self.action_dim, 1)
        diff_ineq_self = torch.zeros((2*self.action_dim,)+self.delta.shape, dtype=torch.float32, device=self.device)
        for i in range(self.action_dim):
            diff_ineq_self[2*i, i] = 1.0
            diff_ineq_self[2*i+1, i] = - 1.0
        self.diff_ineq = torch.cat([self.diff_ineq, diff_ineq_self], dim=0)
        self.diff_ineq[1, :] = - self.diff_ineq[1, :]
        self.diff_ineq_bias = torch.tensor([8, 8, 10, 10, 10, 10], dtype=torch.float32, device=self.device)

        self.diff_eq_np = self.diff_eq.cpu().numpy()
        self.diff_eq_partial_np = self.diff_eq_np[:, self.partial_actions]
        self.diff_eq_other_inv_np= self.diff_eq_other_inv.cpu().numpy()
        self.diff_eq_bias_np = self.diff_eq_bias.cpu().numpy()

        self.diff_ineq_np = self.diff_ineq.cpu().numpy()
        self.diff_ineq_bias_np = self.diff_ineq_bias.cpu().numpy()

        self.holding_eq = False
        self.holding_ineq = False

        self.volatile = False
        self.update = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if not self.action_space.contains(action):
            action_fixed = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action_fixed = action
        assert self.action_space.contains(action_fixed), "%r (%s) invalid" % (action, type(action))

        state = self.state
        x, x_dot, xacc, theta, theta_dot, thetaacc = state
        force_x = action_fixed @ np.cos(self.delta)
        force_y = action_fixed @ np.sin(self.delta)
        force = force_x
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        n_c = force_y + self.total_mass * self.gravity - self.polemass_length * (thetaacc * sintheta + theta_dot * theta_dot*costheta)
        sign = np.sign(n_c * x_dot)
        temp = (force + self.polemass_length * theta_dot * theta_dot *
                (sintheta + self.mu_c * sign * costheta)) / self.total_mass + self.mu_c * self.gravity * sign
        thetaacc = (self.gravity * sintheta - costheta * temp -
                    self.mu_p * theta_dot / self.polemass_length) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * (costheta - self.mu_c*self.gravity*sign) / self.total_mass))
        xacc = (force + self.polemass_length * (theta_dot*theta_dot*sintheta-thetaacc*costheta)
                - self.mu_c * n_c * sign) / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, xacc, theta, theta_dot, thetaacc)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0

        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {'ineq_viol': self.ineq_dist_np(state, action), 'eq_viol': self.eq_resid_np(state, action)}

    def reset(self):
        # self.state = np.zeros((6,))
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        #self.state[4:] = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        # self.state[3] += math.pi
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # self.constr_line_l = rendering.Line((-self.x_constraint * scale + screen_width / 2., carty - 100),
            #                                     (-self.x_constraint * scale + screen_width / 2., carty + 1000))
            # self.constr_line_l.set_color(1, 0, 0)
            # self.viewer.add_geom(self.constr_line_l)
            # self.constr_line_r = rendering.Line((self.x_constraint * scale + screen_width / 2., carty - 100),
            #                                     (self.x_constraint * scale + screen_width / 2., carty + 1000))
            # self.constr_line_r.set_color(1, 0, 0)
            # self.viewer.add_geom(self.constr_line_r)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    ## methods for hard constraint

    def complete_partial(self, state, action_partial):
        action = torch.zeros(state.shape[0], self.action_dim, device=self.device)
        action[:, self.partial_actions] = action_partial
        action[:, self.other_actions] = (self.diff_eq_bias.T - action_partial @ self.diff_eq_partial.T) @ self.diff_eq_other_inv.T
        return action

    def eq_resid(self, state, action):
        return self.diff_eq_bias - action @ self.diff_eq.T

    def ineq_resid(self, state, action):
        return action@self.diff_ineq.T - self.diff_ineq_bias

    def eq_dist(self, state, action):
        resids = self.eq_resid(state, action)
        return torch.abs(resids)

    def ineq_dist(self, state, action):
        resids = self.ineq_resid(state, action)
        return torch.clamp(resids, 0)

    def eq_grad(self, state, action):
        return 2*(action@self.diff_eq.T - self.diff_eq_bias)@self.diff_eq

    def ineq_grad(self, state, action):
        ineq_dist = self.ineq_dist(state, action)
        return 2*ineq_dist@self.diff_ineq

    def ineq_partial_grad(self, state, action, eps=0):
        diff_grad_partial = self.diff_ineq[:, self.partial_actions] - self.diff_ineq[:, self.other_actions] @ (
                    self.diff_eq_other_inv @ self.diff_eq_partial)
        bias_grad_partial = self.diff_ineq_bias - (self.diff_eq_bias @ self.diff_eq_other_inv.T) @ self.diff_ineq[:,
                                                                                                   self.other_actions].T
        bias_modified = torch.clamp(action[:, self.partial_actions] @ diff_grad_partial.T - bias_grad_partial, 0)
        # bias_modified += (bias_modified > 0) * eps
        grad = (bias_modified > 0).to(torch.float32) @ diff_grad_partial
        # grad = 2 * bias_modified @ bias_grad_partial
        action = torch.zeros(state.shape[0], self.action_dim, device=self.device)
        action[:, self.partial_actions] = grad
        action[:, self.other_actions] = - (grad @ self.diff_eq_partial.T) @ self.diff_eq_other_inv.T
        return action

    def ineq_resid_np(self, state, action):
        return action @ self.diff_ineq_np.T - self.diff_ineq_bias_np

    def ineq_dist_np(self, state, action):
        resids = self.ineq_resid_np(state, action)
        return np.clip(resids, 0, None)

    def eq_resid_np(self, state, action):
        return self.diff_eq_bias_np - action @ self.diff_eq_np.T

    def hold_eq(self):
        self.holding_eq = True

    def release_eq(self):
        self.holding_eq = False

    def hold_ineq(self):
        self.holding_ineq = True

    def release_ineq(self):
        self.holding_ineq = False

    @property
    def box_constraint(self):
        return self.action_space.low, self.action_space.high

    @property
    def box_constraint_partial(self):
        return self.action_space.low[self.partial_actions], self.action_space.high[self.partial_actions]

if __name__ == "__main__":
    env = CartSafeEnv()
    for _ in range(10):
        env.reset()
        for i in range(500):
            action = env.action_space.sample()
            print(action)
            obs, reward, done, info = env.step(action)
            print(info)
            env.render()
            if done:
                print(i)
                break
    env.close()