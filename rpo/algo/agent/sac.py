import numpy as np
import torch
import torch.nn as nn
from .base import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
from ..model.dual import DualAdam, Dual
from ..model.policy import GaussianSharedPolicy
from ..model.value import DoubleValueAdd, DoubleValueCat
from ..model.embedding import StateEmbedding, ActionEmbedding
from ...utils.buffer import ReplayBuffer
from ...utils.utils import Type

class PDSAC(Agent):

    def __init__(self, automatic_entropy_tuning, state_dim, action_dim, eq_num, ineq_num, embed_dim=128, hidden_dim=128,
                 hidden_layer=1, shared_param=True, value_type="add", box_constraint=None, alpha=0.2, lr_alpha=1e-4, lr_actor=1e-4,
                 lr_critic=3e-4, lr_dual=1e-4, reg=0, eps=0.1, tau=0.005, gamma=0.98,
                 capacity=10000, ex_action_dim=0,
                 init_lamb=0.0, init_nju=0.0, device=torch.device("cpu")):

        super().__init__(state_dim=state_dim, action_dim=action_dim, box_constraint=box_constraint)

        self.automatic_entropy_tuning = automatic_entropy_tuning

        state_embed = StateEmbedding(state_dim, embed_dim, hidden_dim)
        action_embed1 = ActionEmbedding(action_dim, embed_dim, hidden_dim)
        action_embed2 = ActionEmbedding(action_dim, embed_dim, hidden_dim)
        if shared_param:
            state_policy = state_embed
            state_value1 = state_embed
            state_value2 = state_embed
        else:
            state_policy = state_embed
            state_value1 = StateEmbedding(state_dim, embed_dim, hidden_dim)
            state_value2 = StateEmbedding(state_dim, embed_dim, hidden_dim)

        self.actor = GaussianSharedPolicy(state_dim, action_dim, state_policy, embed_dim, hidden_dim, hidden_layer, box_constraint)
        self.actor.to(device=device)
        if value_type == "add":
            self.critic = DoubleValueAdd(state_dim, action_dim, state_value1, state_value2, action_embed1, action_embed2, embed_dim, hidden_dim)
        elif value_type == "cat":
            self.critic = DoubleValueCat(state_dim, action_dim, state_value1, state_value2, action_embed1, action_embed2, embed_dim, hidden_dim)
        else:
            raise Exception("Unknown Value Net!")
        self.critic.to(device=device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=reg)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=reg)

        self.alpha = alpha
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)


        self.eq_num = eq_num
        self.ineq_num = ineq_num

        self.lamb = Dual(self.eq_num, device=device)
        self.nju = Dual(self.ineq_num, device=device)

        self.lamb.reset_parameters(init_lamb)
        self.nju.reset_parameters(init_nju)

        self.lamb_optim = optim.Adam(self.lamb.parameters(), lr=lr_dual, maximize=True)
        self.nju_optim = DualAdam(self.nju.parameters(), lr=lr_dual)

        self.tau = tau
        self.eps = eps
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity, state=Type((state_dim,), np.float32), action=Type((action_dim,), np.float32), next_state=Type((state_dim,), np.float32), reward=Type((1,), np.float32),
                                          done=Type((1,), np.bool), eq_viol=Type((eq_num,), np.float32), ineq_viol=Type((ineq_num,), np.float32))
        self.device = device

    def add(self, state, action, next_state, reward, done, eq_viol, ineq_viol):
        self.replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done, eq_viol=eq_viol, ineq_viol=ineq_viol)

    def soft_update(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def save_model(self, save_dir):
        torch.save(self.actor.parameters(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.parameters(), os.path.join(save_dir, "critic.pth"))

    def load_model(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth")))
        self.hard_update()

    def take_action(self, state, deterministic=False, log_pi=False):

        state = torch.tensor(state, device=self.device)
        action_partial, log_std, mean = self.actor(state)
        if deterministic:
            action_partial = mean
        if log_pi:
            return action_partial, log_std
        else:
            return action_partial

    def actor_log_std(self, state):

        state = torch.tensor(state, device=self.device)
        action_partial, log_std, mean = self.actor(state)
        return log_std

    def random_action(self, x):
        return self.box_constraint.sample(x)

    def eps_decay(self, decay_value, lb):
        self.eps = max(lb, self.eps-decay_value)


