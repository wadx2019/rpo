import copy

import torch
import torch.nn.functional as F
from rpo.algo.agent import PDDDPG_PA
from .model.utils import BoxConstraint
from rpo.utils.utils import max_grad
import numpy as np
import os


class RPODDPG(object):

    def __init__(self, env, work_dir, name, logger, max_steps=10, embed_dim=256, hidden_dim=256,
                 hidden_layer=1, shared_param=True, value_type="add", ex_action_dim=0,
                 lr_actor=1e-4, lr_critic=3e-4, lr_dual=1e-4, reg=0, eps=0.1, eps_start=1.0,
                 eps_epoch=10000, tau=0.005, gamma=0.95, capacity=10000,
                 warmup=1000, corr_lr=1e-5, eval_lr=1e-5, corr_mode=0,
                 corr_eps=1e-5, corr_momentum=0.5, batch_size=256,
                 policy_fre=2, eval_fre=500, max_epochs=100000, grad_eps=1e-3, eval_steps=None,
                 init_lamb=0.0, init_nju=0.0, fixed=False, clip_thres="inf", partial=False, partial_idx=None,
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):

        self.env = env
        self.agent = PDDDPG_PA(
                            env.state_dim, env.action_dim, env.eq_num, env.ineq_num,
                            embed_dim=embed_dim, hidden_dim=hidden_dim, hidden_layer=hidden_layer,
                            shared_param=shared_param, value_type=value_type, ex_action_dim=ex_action_dim, box_constraint=BoxConstraint(*self.env.box_constraint_partial, device=device, volatile=self.env.volatile, update=self.env.update),
                            lr_actor=lr_actor, lr_critic=lr_critic, lr_dual=lr_dual,
                            reg=reg, eps=eps_start, tau=tau, gamma=gamma, capacity=capacity,
                            init_lamb=init_lamb, init_nju=init_nju, partial=partial, partial_idx=env.partial_actions if partial_idx is None else partial_idx,
                            device=device
                            )

        self.max_steps = max_steps
        self.corr_lr = corr_lr
        self.eval_lr = eval_lr
        self.corr_eps = corr_eps
        self.corr_momentum = corr_momentum
        self.corr_mode = corr_mode
        self.grad_eps = grad_eps

        self.clip_thres = clip_thres
        if eval_steps is None:
            self.eval_steps = max_steps
        else:
            self.eval_steps = eval_steps

        self.batch_size = batch_size
        self.policy_fre = policy_fre
        self.eval_fre = eval_fre

        self.device = device
        self.work_dir = work_dir
        self.warmup = warmup
        self.max_epochs = max_epochs

        self.fixed = fixed
        self.partial = partial

        self.env_eval = copy.deepcopy(self.env)

        self.box_constraint = BoxConstraint(*self.env.box_constraint, device=device)

        self.name = name

        self.logger = logger

        self.eps = eps
        self.decay_value = (eps_start-eps)/eps_epoch

    def process_action(self, state, action_partial, train=True):

        action = self.env.complete_partial(state, action_partial)
        action = self.grad_steps(state, action, train=train)

        return action

    def run(self, logger=None, eval=True):

        t = 0
        episode_t = 0
        episode_reward = 0
        episode_max_ineq_viol = 0
        episode_max_eq_viol = 0
        n_episode = 0

        state = self.env.reset()
        max_ineqs = []
        max_eqs = []
        while t < self.max_epochs:

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = state.unsqueeze(0)

            with torch.no_grad():
                if t < self.warmup:
                    action_partial = self.agent.random_action(state)
                    action_partial.unsqueeze(0)
                    action = self.process_action(state, action_partial)

                else:

                    action_partial = self.agent.take_action(state)
                    # print("action:", action_partial)
                    action = self.process_action(state, action_partial)

            action = action.squeeze(0)
            action = action.cpu().numpy()

            self.agent.eps_decay(self.decay_value, self.eps)

            next_state, reward, done, info = self.env.step(action)

            eq_viol = info['eq_viol']
            ineq_viol = info['ineq_viol']

            self.agent.add(state.squeeze(0).cpu().numpy(), action, next_state, reward, done, eq_viol, ineq_viol)

            max_eq = np.abs(eq_viol).max()
            max_ineq = ineq_viol.max()
            max_eqs.append(max_eq)
            max_ineqs.append(max_ineq)

            episode_max_eq_viol = max(max_eq, episode_max_eq_viol)
            episode_max_ineq_viol = max(max_ineq, episode_max_ineq_viol)

            t += 1
            episode_t += 1
            episode_reward += reward

            if done:

                print(f"episode {t} ends. reward: {episode_reward}, step: {episode_t}, ineq_viol: {episode_max_ineq_viol}, eq_viol: {episode_max_eq_viol}")
                steps = len(max_ineqs)
                for i in range(steps):
                    self.logger.add(epoch=t-steps+i, reward=episode_reward, max_ineq=max_ineqs[i], max_eq=max_eqs[i])
                episode_t = 0
                episode_reward = 0
                episode_max_eq_viol = 0
                episode_max_ineq_viol = 0
                next_state = self.env.reset()
                n_episode += 1
                max_ineqs = []
                max_eqs = []

            if t % self.eval_fre == 0 and t > self.warmup and eval:
                rmean, rstd, ineqmean, ineqstd, eqmean, eqstd, maxineqmean, maxineqstd,\
                    maxeqmean, maxeqstd = self.eval()
                print()
                print("============================")
                print(f"Eval: epoch {t}, rewards: {rmean:.4f}({rstd:.4f}), mean_ineq_viol: {ineqmean:.4f}({ineqstd:.4f}),"+
                      f" mean_eq_viol: {eqmean:.4f}({eqstd:.4f}), max_ineq_viol: {maxineqmean:.4f}({maxineqstd:.4f})"+
                      f" max_eq_viol: {maxeqmean:.4f}({maxeqstd:.4f})")
                print(f"lambda: {self.agent.lamb}, nju: {self.agent.nju}")
                print("============================")
                print()

            state = next_state
            if t >= self.warmup:
                self.train(t)

    def train(self, t):

        samples = self.agent.replay_buffer.sample(self.batch_size)
        state, action, next_state, reward, done, ineq_viol, eq_viol = samples['state'], samples['action'], samples['next_state'], samples['reward'], samples['done'], samples['ineq_viol'], samples['eq_viol']

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        ineq_viol = torch.tensor(ineq_viol, dtype=torch.float32, device=self.device)
        eq_viol = torch.tensor(eq_viol, dtype=torch.float32, device=self.device)


        critic_loss = self.critic_loss(state, action, next_state, done, reward, ineq_viol, eq_viol)
        self.agent.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.clip_thres, "inf")
        max_grad_critic = max_grad(self.agent.critic)
        self.agent.critic_optim.step()

        if t % self.policy_fre == 0:

            actor_loss = self.actor_loss(state)

            self.agent.actor_optim.zero_grad()
            self.agent.lamb_optim.zero_grad()
            self.agent.nju_optim.zero_grad()

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.clip_thres, "inf")

            if t % 60 == 0:
                print("max_gradient_actor: ", max_grad(self.agent.actor))
                print("max_gradient_critic: ", max_grad_critic)

            self.agent.actor_optim.step()

            if not self.fixed:
                # self.agent.lamb_optim.step()
                self.agent.nju_optim.step()

            self.agent.soft_update()

    def eval(self, rendering=False):

        nn = 0
        pro = 0

        total_rewards = []
        mean_ineq_viols = []
        mean_eq_viols = []
        max_ineq_viols = []
        max_eq_viols = []
        for _ in range(10):
            total_reward = 0
            mean_ineq_viol = 0
            mean_eq_viol = 0
            max_ineq_viol = 0
            max_eq_viol = 0
            state = self.env_eval.reset()

            for i in range(500):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                state = state.unsqueeze(0)
                with torch.no_grad():
                    import time
                    start = time.time()
                    action_partial = self.agent.take_action(state, deterministic=True)
                    median = time.time()
                    action, step = self.process_action(state, action_partial, train=False)
                    end = time.time()
                    nn += median - start
                    pro += end - median
                action = action.squeeze(0).cpu().numpy()
                state, reward, done, info = self.env_eval.step(action)
                total_reward += reward
                mean_ineq_viol += (info["ineq_viol"].max()-mean_ineq_viol) / (i+1)
                mean_eq_viol += (np.abs(info["eq_viol"]).max() - mean_eq_viol) / (i + 1)
                max_ineq_viol = max(max_ineq_viol, info["ineq_viol"].max())
                max_eq_viol = max(max_eq_viol, np.abs(info["eq_viol"]).max())
                if rendering:
                    self.env_eval.render()
                if done:
                    break

            total_rewards.append(total_reward)
            mean_ineq_viols.append(mean_ineq_viol)
            mean_eq_viols.append(mean_eq_viol)
            max_ineq_viols.append(max_ineq_viol)
            max_eq_viols.append(max_eq_viol)

        print("nn:", nn, "pro:", pro, pro/nn)
        total_rewards = np.array(total_rewards)
        mean_ineq_viols = np.array(mean_ineq_viols)
        mean_eq_viols = np.array(mean_eq_viols)
        max_ineq_viols = np.array(max_ineq_viols)
        max_eq_viols = np.array(max_eq_viols)

        return total_rewards.mean(), total_rewards.std(), mean_ineq_viols.mean(), mean_ineq_viols.std(),\
               mean_eq_viols.mean(), mean_eq_viols.std(), max_ineq_viols.mean(), max_ineq_viols.std(), \
               max_eq_viols.mean(), max_eq_viols.std()

    def grad_steps(self, state, action, train=True):

        step = 0
        old_action_step = 0
        if train:
            while (step == 0 or torch.max(torch.abs(self.env.eq_resid(state, action))) > self.corr_eps or
                   torch.max(self.env.ineq_dist(state, action)) > self.corr_eps) and step < self.max_steps:
                if self.corr_mode == 0:
                    action_step = self.env.ineq_partial_grad(state, action, eps=self.grad_eps)
                else:
                    ineq_step = self.env.ineq_grad(state, action)
                    eq_step = self.env.eq_grad(state, action)
                    action_step = self.agent.nju(ineq_step) + self.agent.lamb(eq_step)

                action_step = self.corr_lr * action_step + self.corr_momentum * old_action_step
                action = action - action_step

                old_action_step = action_step
                step += 1

            return action

        else:
            with torch.no_grad():
                while (step == 0 or torch.max(torch.abs(self.env.eq_resid(state, action))) > self.corr_eps or
                       torch.max(self.env.ineq_dist(state, action)) > self.corr_eps) and step < self.eval_steps:
                    if self.corr_mode == 0:
                        action_step = self.env.ineq_partial_grad(state, action, eps=self.grad_eps)
                    else:
                        ineq_step = self.env.ineq_grad(state, action)
                        eq_step = self.env.eq_grad(state, action)
                        action_step = self.agent.nju(ineq_step) + self.agent.lamb(eq_step)

                    action_step = self.eval_lr * action_step + self.corr_momentum * old_action_step
                    action = action - action_step

                    old_action_step = action_step
                    step += 1

            return action, step

    def actor_loss(self, state):

        action_partials = self.agent.take_action(state)
        actions = self.env.complete_partial(state, action_partials)
        # ineq_viol = self.env.ineq_resid(state, actions)
        ineq_viol = self.env.ineq_dist(state, actions)
        eq_viol = self.env.eq_resid(state, actions)

        # actions = self.grad_steps(state, actions)
        loss = -self.agent.critic(state, actions)
        # loss = -self.env.obj_fn(actions).view(-1,1)

        loss += self.agent.nju(ineq_viol)
        # loss += self.agent.lamb(torch.abs(eq_viol))

        loss = loss.mean()

        return loss


    def critic_loss(self, state, action, next_state, done, reward, ineq_viol=None, eq_viol=None):
        with torch.no_grad():
            next_action_partials = self.agent.take_action(next_state, deterministic=True, target=True)
            next_actions = self.process_action(next_state, next_action_partials)
            next_q_values = self.agent.critic_target(next_state, next_actions)
            target_q_values = reward + self.agent.gamma * (1 - done) * next_q_values

        q_values = self.agent.critic(state, action)
        loss = F.smooth_l1_loss(q_values, target_q_values)

        return loss

    def load(self):
        self.agent.load_model(self.work_dir)

    def save(self):
        self.agent.save_model(self.work_dir)
