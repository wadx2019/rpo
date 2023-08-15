import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class StateEmbedding(nn.Module):

    def __init__(self, state_dim, embed_dim, hidden_dim=256, embed_layer=1):
        super().__init__()
        self.embeds = nn.ModuleList()
        if embed_layer == 1:
            hidden_dim = embed_dim
        self.embeds.append(nn.Linear(state_dim, hidden_dim))

        for i in range(embed_layer-1):
            if i == embed_layer-1:
                self.embeds.append(nn.Linear(hidden_dim, embed_dim))
            else:
                self.embeds.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, s):

        for i, embed in enumerate(self.embeds):
            if i == 0:
                s = embed(s)
            else:
                s = embed(F.relu(s))

        return s

class ActionEmbedding(nn.Module):

    def __init__(self, action_dim, embed_dim, hidden_dim=256, embed_layer=1):
        super().__init__()
        self.embeds = nn.ModuleList()
        if embed_layer == 1:
            hidden_dim = embed_dim
        self.embeds.append(nn.Linear(action_dim, hidden_dim))

        for i in range(embed_layer-1):
            if i == embed_layer-1:
                self.embeds.append(nn.Linear(hidden_dim, embed_dim))
            else:
                self.embeds.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, a):

        for i, embed in enumerate(self.embeds):
            if i == 0:
                a = embed(a)
            else:
                a = embed(F.relu(a))

        return a


class SharedEmbedding(nn.Module):

    def __init__(self, embed_dim_in, embed_dim_out, hidden_dim=256, embed_layer=1):
        super().__init__()
        self.embeds = nn.ModuleList()
        if embed_layer == 1:
            hidden_dim = embed_dim_out
        self.embeds.append(nn.Linear(embed_dim_in, hidden_dim))

        for i in range(embed_layer - 1):
            if i == embed_layer - 1:
                self.embeds.append(nn.Linear(hidden_dim, embed_dim_out))
            else:
                self.embeds.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for i, embed in enumerate(self.embeds):
            if i == 0:
                x = embed(x)
            else:
                x = embed(F.relu(x))
        return x



