import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = -2
LOG_SIG_MIN = -23

class SharedPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, state_embed, embed_dim, hidden_dim=256, hidden_layer=1, box_constraint=None):

        super().__init__()
        self.box_constraint = box_constraint
        self.state_embed = state_embed
        self.affines = nn.ModuleList()
        self.affines.append(nn.Linear(embed_dim, hidden_dim))
        for i in range(hidden_layer):
            if i == hidden_layer-1:
                self.affines.append(nn.Linear(hidden_dim, action_dim))
            else:
                self.affines.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, s):

        x = self.state_embed(s)
        for affine in self.affines:
            x = affine(F.relu(x))

        if self.box_constraint:
            x = self.box_constraint(F.tanh(x), s)

        return x

class GaussianSharedPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, state_embed, embed_dim, hidden_dim=256, hidden_layer=1, box_constraint=None):

        super().__init__()
        self.box_constraint = box_constraint
        self.state_embed = state_embed
        self.affines = nn.ModuleList()
        self.affines.append(nn.Linear(embed_dim, hidden_dim))
        for i in range(hidden_layer-1):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affine_mean = nn.Linear(hidden_dim, action_dim)
        self.affine_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, s):

        x = self.state_embed(s)
        for affine in self.affines:
            x = affine(F.relu(x))
        x = F.relu(x)
        mean = self.affine_mean(x)
        log_std = self.affine_log_std(x) - 3
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        log_prob = normal.log_prob(x)
        if self.box_constraint:
            y = F.tanh(x)
            x = self.box_constraint(y, s)
            mean = self.box_constraint(F.tanh(mean), s)
            log_prob -= torch.log(self.box_constraint.action_scale(s)*(1-y.pow(2))+1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob -= torch.log(1 - x.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

        return x, log_prob, mean
