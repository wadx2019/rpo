import numpy as np
import torch
import torch.nn as nn

class BoxConstraint(object):

    def __init__(self, cmin, cmax, device, style="tanh", volatile=False, update=None, full=False):

        self.style = style
        self.cmin = cmin
        self.cmax = cmax
        self.volatile = volatile
        self.update = update
        self.full = full

        if style == "sigmoid":
            self.base = cmin
            self.scale = cmax-cmin
        elif style == "tanh":
            self.scale = (cmax-cmin) / 2
            self.base = cmin + self.scale

        # for torch version
        self.device = device
        self.base_torch = torch.tensor(self.base, dtype=torch.float32).to(device)
        self.scale_torch = torch.tensor(self.scale, dtype=torch.float32).to(device)
        self.cmin_torch = torch.tensor(self.cmin, dtype=torch.float32).to(device)
        self.cmax_torch = torch.tensor(self.cmax, dtype=torch.float32).to(device)

        print("cmax:", cmax, "cmin", cmin, "scale", self.scale, "base", self.base)

    def cuda(self):
        self.base_torch = self.base_torch.cuda()
        self.scale_torch = self.scale_torch.cuda()

    def to(self, *args, **kwargs):
        self.base_torch = self.base_torch.to(*args, **kwargs)
        self.scale_torch = self.scale_torch.to(*args, **kwargs)

    def __call__(self, x, state=None):
        if self.volatile:
            self.update_box(state)
            if type(x) == torch.Tensor:
                return self.scale_torch_vol * x + self.base_torch_vol
            else:
                return self.scale_vol * x + self.base_vol
        else:
            if type(x) == torch.Tensor:
                return self.scale_torch * x + self.base_torch
            else:
                return self.scale * x + self.base

    def sample(self, state):
        tmp = torch.rand_like(self.base_torch)
        if self.style == "tanh":
            tmp = 2 * tmp - 1

        if self.volatile:
            self.update_box(state)
            return self.scale_torch_vol * tmp + self.base_torch_vol
        else:
            return self.scale_torch * tmp + self.base_torch

    def sample_np(self, state):
        tmp = np.random.rand(*self.base.shape)
        if self.style == "tanh":
            tmp = 2 * tmp - 1

        if self.volatile:
            self.update_box(state)
            return self.scale_vol * tmp + self.base_vol
        else:
            return self.scale * tmp + self.base

    def update_box(self, x):
        self.cmin_vol, self.cmax_vol = self.update(x, full=self.full)
        # print(cmin_vol, cmax_vol)
        if self.style == "sigmoid":
            self.base_vol = self.cmin_vol
            self.scale_vol = self.cmax_vol - self.cmin_vol
        elif self.style == "tanh":
            self.scale_vol = (self.cmax_vol - self.cmin_vol) / 2
            self.base_vol = self.cmin_vol + self.scale_vol

        self.base_torch_vol = torch.tensor(self.base_vol, dtype=torch.float32).to(self.device)
        self.scale_torch_vol = torch.tensor(self.scale_vol, dtype=torch.float32).to(self.device)
        self.cmax_torch_vol = torch.tensor(self.cmax_vol, dtype=torch.float32).to(self.device)
        self.cmin_torch_vol = torch.tensor(self.cmin_vol, dtype=torch.float32).to(self.device)

    def clip(self, x, state=None):
        if self.volatile:
            self.update_box(state)
            if type(x) == torch.Tensor:
                return torch.clip(x, self.cmin_torch_vol, self.cmax_torch_vol)
            else:
                return np.clip(x, self.cmin_vol, self.cmax_vol)
        else:
            if type(x) == torch.Tensor:
                return torch.clip(x, self.cmin_torch, self.cmax_torch)
            else:
                return np.clip(x, self.cmin, self.cmax)

    def action_scale(self, state=None):
        if self.volatile:
            self.update_box(state)
            if type(state) == torch.Tensor:
                return self.scale_torch_vol
            else:
                return self.scale_vol
        else:
            if type(state) == torch.Tensor:
                return self.scale_torch
            else:
                return self.scale
