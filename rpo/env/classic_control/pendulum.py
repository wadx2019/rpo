import gym
from gym import spaces
from gym.utils import seeding
from gym.error import DependencyNotInstalled
import numpy as np
from os import path
import torch


class SpringPendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.max_speed = 8.0
        self.max_torque = 6.0
        self.max_length_delta = 0.5

        self.max_summation = 32.0

        self.dt = 0.05
        self.g = g
        self.m = 0.5
        self.viewer = None
        self.k = 1.0
        self.l0 = 1.0

        self.m_dt = self.m / self.dt


        high = np.array([1.0, 1.0, self.max_speed, 1.5, 0.05], dtype=np.float32)
        low = np.array([-1.0, -1.0, -self.max_speed, 0.5, -0.05], dtype=np.float32)
        action_high = self.max_torque * np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-action_high, high=action_high, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.eq_num = 1
        self.ineq_num = 1
        self.fx_idx = 0
        self.fy_idx = 1

        self.seed()

        self.partial_actions = np.array([0])
        self.other_actions = np.setdiff1d(np.arange(self.action_dim), self.partial_actions)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.define_obs_idx()

        self.diff_eq_partial = None
        self.diff_eq_other_inv = None
        self.diff_eq = None
        self.diff_eq_bias = None

        self.diff_ineq = None
        self.diff_ineq_bias = torch.tensor([self.max_summation], dtype=torch.float32, device=self.device)


        self.holding_eq = False
        self.holding_ineq = False

        self.counter = None

        self.volatile = False
        self.update = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        state = self._get_obs()
        self.counter += 1

        if not self.action_space.contains(action):
            action_fixed = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action_fixed = action
        assert self.action_space.contains(action_fixed), "%r (%s) invalid" % (action, type(action))


        th, thdot, l, ldot = self.state  # th := theta

        fx, fy = action_fixed
        fth = -fy * np.sin(th) + fx * np.cos(th)
        fl = fy * np.cos(th) + fx * np.sin(th)

        g = self.g
        m = self.m
        dt = self.dt
        l0 = self.l0
        k = self.k

        u = np.clip(fth, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        costs = np.abs(angle_normalize(th))

        #semi-implicit euler
        thacc = (fth - m * (g * np.sin(th) + 2 * ldot * thdot)) / (l * m)
        lacc = (fl - m * g * np.cos(th) + m * l * thdot ** 2 - k * (l - l0)) / m
        newthdot = thdot + thacc * dt
        newldot = ldot + lacc * dt
        # eq_viol = self.eq_resid_np(state, action)
        # if eq_viol >= 0.01:
        #     print("newldot", newldot, action, state, self.counter)
        #     print("eq_viol", eq_viol)
        #     raise Exception
        # raise Exception
        newth = th + newthdot * dt
        newl = l + ldot * dt

        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        done = (newl <= 0.5 or newl >= 1.5 or newth >= np.pi/12 or newth <= -np.pi/12)

        self.state = np.array([newth, newthdot, newl, newldot])

        return self._get_obs(), 1 / (100 * costs + 1), done, {'ineq_viol': self.ineq_dist_np(state, action), 'eq_viol': self.eq_resid_np(state, action)}

    def reset(self):
        high = np.array([np.pi/12, 1.0, 1.05, 0.05], dtype=np.float32)
        low = np.array([-np.pi/12, -1.0, 0.95, -0.05], dtype=np.float32)
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        self.counter = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot, l, ldot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot, l, ldot])

    def define_obs_idx(self):
        self.idx_cos, self.idx_sin, self.idx_thdot, self.idx_l, self.idx_ldot = np.arange(5)

    def render(self, mode="human"):
        self.render_mode = "rgb_array"
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        # change this part to change the length of spring
        # rod_length = 1 * scale
        rod_length = self.state[1] * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)

        # replace thie part with spring
        # gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        # gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        # pygame.draw.aaline(self.surf, (204, 77, 77),
        #                    (transformed_coords[0][0], (transformed_coords[0][1] + transformed_coords[1][1]) / 2),
        #                    (transformed_coords[2][0], (transformed_coords[2][1] + transformed_coords[3][1]) / 2))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 10), (0, 0, 0))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 10), (0, 0, 0)
        )

        spring_coords = self.generate_spring_coordinate(transformed_coords)
        # print(spring_coords)
        pygame.draw.aalines(self.surf, (0, 0, 0), False, spring_coords)

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (255, 0, 0)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (255, 0, 0)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        # gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        # gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def generate_spring_coordinate(self, coords):
        num_fold = 70
        frac_line = 0.15
        spring_coords = []
        start_point = (coords[0][0], (coords[0][1] + coords[1][1]) / 2)
        end_point = (coords[2][0], (coords[2][1] + coords[3][1]) / 2)
        start_spring_point = ((1 - frac_line) * start_point[0] + frac_line * end_point[0],
                              (1 - frac_line) * start_point[1] + frac_line * end_point[1])
        end_spring_point = (frac_line * start_point[0] + (1 - frac_line) * end_point[0],
                            frac_line * start_point[1] + (1 - frac_line) * end_point[1])
        coord_a = ((1 - frac_line) * coords[3][0] + frac_line * coords[0][0],
                   (1 - frac_line) * coords[3][1] + frac_line * coords[0][1])
        coord_b = (frac_line * coords[3][0] + (1 - frac_line) * coords[0][0],
                   frac_line * coords[3][1] + (1 - frac_line) * coords[0][1])
        coord_c = ((1 - frac_line) * coords[2][0] + frac_line * coords[1][0],
                   (1 - frac_line) * coords[2][1] + frac_line * coords[1][1])
        coord_d = (frac_line * coords[2][0] + (1 - frac_line) * coords[1][0],
                   frac_line * coords[2][1] + (1 - frac_line) * coords[1][1])
        spring_coords.append(start_point)
        spring_coords.append(start_spring_point)
        for i in range(1, num_fold, 2):
            if ((i - 1) / 2) % 2 == 0:
                point = ((i / num_fold) * coord_a[0] + ((num_fold - i) / num_fold) * coord_b[0],
                         (i / num_fold) * coord_a[1] + ((num_fold - i) / num_fold) * coord_b[1])
            else:
                point = ((i / num_fold) * coord_c[0] + ((num_fold - i) / num_fold) * coord_d[0],
                         (i / num_fold) * coord_c[1] + ((num_fold - i) / num_fold) * coord_d[1])
            spring_coords.append(point)
        spring_coords.append(end_spring_point)
        spring_coords.append(end_point)

        return spring_coords

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def complete_partial(self, state, action_partial):
        self.set_eq(state, action_partial)
        action = torch.zeros(state.shape[0], self.action_dim, device=self.device)
        action[:, self.partial_actions] = action_partial
        action[:, self.other_actions] = torch.bmm(self.diff_eq_bias.view(-1, 1, 1) - torch.bmm(action_partial.view(-1, 1, 1), self.diff_eq_partial.view(-1, 1, 1)), self.diff_eq_other_inv.view(-1, 1, 1)).view(-1, 1)
        # print("action", action, "diff_bias", self.diff_eq_bias, self.diff_eq_partial, self.diff_eq_other_inv)
        return action

    def set_eq(self, state, action_partial):

        if self.holding_eq:
            return

        costheta = state[:, self.idx_cos]
        sintheta = state[:, self.idx_sin]
        thetadot = state[:, self.idx_thdot]
        l = state[:, self.idx_l]
        ldot = state[:, self.idx_ldot]

        if self.fx_idx in self.partial_actions:
            diff_eq_partial = torch.tensor(sintheta, dtype=torch.float32, device=self.device).view(-1, 1)
            diff_eq_other = torch.tensor(costheta, dtype=torch.float32, device=self.device).view(-1, 1)
            self.diff_eq = torch.cat([diff_eq_partial, diff_eq_other], dim=1)
        else:
            diff_eq_partial = torch.tensor(costheta, dtype=torch.float32, device=self.device).view(-1, 1)
            diff_eq_other = torch.tensor(sintheta, dtype=torch.float32, device=self.device).view(-1, 1)
            self.diff_eq = torch.cat([diff_eq_other, diff_eq_partial], dim=1)

        self.diff_eq_partial = diff_eq_partial
        self.diff_eq_other_inv = 1 / diff_eq_other

        diff_eq_bias = - self.m_dt * ldot - (l * self.m * thetadot**2 - self.k * (l - self.l0) - self.m * self.g * costheta)
        self.diff_eq_bias = torch.tensor(diff_eq_bias, dtype=torch.float32, device=self.device).view(-1, 1)

        # print("diffbias:", diff_eq_bias, state)

    def set_ineq(self, state, action):
        if self.holding_ineq:
            return

        self.diff_ineq = 2 * action.detach().clone()

    def eq_resid(self, state, action):
        self.set_eq(state, action)
        return self.diff_eq_bias - torch.bmm(action.view(-1, 1, self.action_dim), self.diff_eq.view(-1, self.action_dim, 1)).view(-1, 1)

    def ineq_resid(self, state, action):
        return torch.bmm(action.view(-1, 1, self.action_dim), action.view(-1, self.action_dim, 1)).view(-1, 1) - self.diff_ineq_bias

    def eq_dist(self, state, action):
        resids = self.eq_resid(state, action)
        return torch.abs(resids)

    def ineq_dist(self, state, action):
        resids = self.ineq_resid(state, action)
        return torch.clamp(resids, 0)

    def ineq_dist_np(self, state, action):
        action = torch.tensor(action, device=self.device).view(1, -1)
        state = torch.tensor(state, device=self.device).view(1, -1)
        return self.ineq_dist(state, action).detach().cpu().numpy()

    def eq_resid_np(self, state, action):
        action = torch.tensor(action, device=self.device).view(1, -1)
        state = torch.tensor(state, device=self.device).view(1, -1)
        return self.eq_resid(state, action).detach().cpu().numpy()

    def eq_grad(self, state, action):
        return 2 * (action @ self.diff_eq.T - self.diff_eq_bias) @ self.diff_eq

    def ineq_grad(self, state, action):
        self.set_ineq(state, action)
        ineq_dist = self.ineq_dist(state, action)
        return 2 * self.diff_ineq.transpose(1, 2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    def ineq_partial_grad(self, state, action, eps=0):
        self.set_ineq(state, action)
        self.set_eq(state, action)
        diff_grad_partial = self.diff_ineq[:, self.partial_actions] - self.diff_ineq[:, self.other_actions].view(-1,1,1).bmm(
                    self.diff_eq_other_inv.view(-1,1,1).bmm(self.diff_eq_partial.view(-1,1,1))).view(-1,1)
        bias_grad_partial = self.diff_ineq_bias - (self.diff_eq_bias.view(-1,1,1).bmm(self.diff_eq_other_inv.view(-1,1,1)).bmm(self.diff_ineq[:, self.other_actions].view(-1,1,1))).view(-1,1)
        bias_modified = torch.clamp(action[:, self.partial_actions] @ diff_grad_partial.T - bias_grad_partial, 0)
        # bias_modified += (bias_modified > 0) * eps
        grad = (bias_modified > 0).to(torch.float32) @ diff_grad_partial
        action = torch.zeros(state.shape[0], self.action_dim, device=self.device)
        action[:, self.partial_actions] = grad
        action[:, self.other_actions] = - (grad.view(-1,1,1).bmm(self.diff_eq_partial.view(-1,1,1))).bmm(self.diff_eq_other_inv.view(-1,1,1)).view(-1,1)
        return action

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


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

