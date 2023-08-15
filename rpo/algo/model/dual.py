import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from typing import List, Optional
from torch.optim.optimizer import Optimizer, required

class DualSGD(optim.SGD):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, foreach: Optional[bool] = None):

        super(DualSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                 weight_decay=weight_decay, nesterov=nesterov, maximize=True, foreach=foreach)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)

        for group in self.param_groups:
            for p in group["params"]:
                p.clamp_(0)

        return loss

class DualAdam(optim.Adam):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 capturable: bool = False):

        super(DualAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                 weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach,
                 maximize=True, capturable=capturable)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)

        for group in self.param_groups:
            for p in group["params"]:
                p.clamp_(0)

        return loss

class Dual(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dim = dim
        self.weight = nn.Parameter(torch.empty((1, dim),  **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self, value=0.0):

        init.constant_(self.weight, value)

    def __repr__(self):

        return f"Dual(weight: {self.weight})"

    def forward(self, x):

        return F.linear(x, self.weight)



