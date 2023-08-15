import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedValueCat(nn.Module):

    def __init__(self, state_dim, action_dim, state_embed, action_embed, embed_dim, hidden_dim=256, hidden_layer=1,
                 partial=False, partial_idx=None):

        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        self.partial = partial
        self.partial_idx = partial_idx
        self.affines = nn.ModuleList()
        self.affines.append(nn.Linear(embed_dim * 2, hidden_dim))
        for i in range(hidden_layer):
            if i == hidden_layer-1:
                self.affines.append(nn.Linear(hidden_dim, 1))
            else:
                self.affines.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, s, a):
        if self.partial:
            a = a[:, self.partial_idx]
        s = self.state_embed(s)
        a = self.action_embed(a)
        x = torch.cat([s, a], dim=1)
        for affine in self.affines:
            x = affine(F.relu(x))
        return x

class SharedValueAdd(nn.Module):

    def __init__(self, state_dim, action_dim, state_embed, action_embed, embed_dim, hidden_dim=256, hidden_layer=1,
                 partial=False, partial_idx=None):

        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        self.partial = partial
        self.partial_idx = partial_idx
        self.affines = nn.ModuleList()
        self.affines.append(nn.Linear(embed_dim, hidden_dim))
        for i in range(hidden_layer):
            if i == hidden_layer-1:
                self.affines.append(nn.Linear(hidden_dim, 1))
            else:
                self.affines.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, s, a):
        if self.partial:
            a = a[:, self.partial_idx]
        s = self.state_embed(s)
        a = self.action_embed(a)
        x = s + a
        for affine in self.affines:
            x = affine(F.relu(x))
        return x

class DoubleValueCat(nn.Module):

    def __init__(self, state_dim, action_dim, state_embed1, state_embed2, action_embed1, action_embed2, embed_dim, hidden_dim=256, hidden_layer=1,
                 partial=False, partial_idx=None):

        super().__init__()
        self.state_embed1 = state_embed1
        self.action_embed1 = action_embed1
        self.state_embed2 = state_embed2
        self.action_embed2 = action_embed2
        self.partial = partial
        self.partial_idx = partial_idx
        self.affines1 = nn.ModuleList()
        self.affines1.append(nn.Linear(embed_dim * 2, hidden_dim))
        self.affines2 = nn.ModuleList()
        self.affines2.append(nn.Linear(embed_dim * 2, hidden_dim))
        for i in range(hidden_layer):
            if i == hidden_layer-1:
                self.affines1.append(nn.Linear(hidden_dim, 1))
                self.affines2.append(nn.Linear(hidden_dim, 1))
            else:
                self.affines1.append(nn.Linear(hidden_dim, hidden_dim))
                self.affines2.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, s, a):
        if self.partial:
            a = a[:, self.partial_idx]
        s1 = self.state_embed1(s)
        a1 = self.action_embed1(a)
        x1 = torch.cat([s1, a1], dim=1)
        for affine in self.affines1:
            x1 = affine(F.relu(x1))

        s2 = self.state_embed2(s)
        a2 = self.action_embed2(a)
        x2 = torch.cat([s2, a2], dim=1)
        for affine in self.affines2:
            x2 = affine(F.relu(x2))

        return x1, x2

class DoubleValueAdd(nn.Module):

    def __init__(self, state_dim, action_dim, state_embed1, state_embed2, action_embed1, action_embed2, embed_dim, hidden_dim=256, hidden_layer=1,
                 partial=False, partial_idx=None):
        super().__init__()
        self.state_embed1 = state_embed1
        self.action_embed1 = action_embed1
        self.state_embed2 = state_embed2
        self.action_embed2 = action_embed2
        self.partial = partial
        self.partial_idx = partial_idx
        self.affines1 = nn.ModuleList()
        self.affines1.append(nn.Linear(embed_dim, hidden_dim))
        self.affines2 = nn.ModuleList()
        self.affines2.append(nn.Linear(embed_dim, hidden_dim))
        for i in range(hidden_layer):
            if i == hidden_layer-1:
                self.affines1.append(nn.Linear(hidden_dim, 1))
                self.affines2.append(nn.Linear(hidden_dim, 1))
            else:
                self.affines1.append(nn.Linear(hidden_dim, hidden_dim))
                self.affines2.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, s, a):
        if self.partial:
            a = a[:, self.partial_idx]
        s1 = self.state_embed1(s)
        a1 = self.action_embed1(a)
        x1 = s1 + a1
        for affine in self.affines1:
            x1 = affine(F.relu(x1))

        s2 = self.state_embed2(s)
        a2 = self.action_embed2(a)
        x2 = s2 + a2
        for affine in self.affines2:
            x2 = affine(F.relu(x2))

        return x1, x2
