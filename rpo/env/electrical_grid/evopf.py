import copy

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import time

from pypower.api import case14
from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

from copy import deepcopy
import os

from .data import DemandLoader, PriceLoader

import igraph

class Battery(object):

    def __init__(self, p_data, num, genbase, low=0.1, high=0.8, p_min=-0.2, p_max=0.2,
                 eta_in=0.9, eta_out=0.9, init_strategy="full", residual=1.0, device=None):
        """
        :param num:
        :param p_min:
        :param p_max:
        :param eta_in:
        :param eta_out:
        :param init_strategy: "full" or "empty" or "random"
        """
        self.p_data = p_data
        self.num = num
        self.genbase = genbase
        self.low = low
        self.high = high
        self.residual = residual

        self.p_min = p_min
        self.p_max = p_max

        self.eta_in = eta_in
        self.eta_out = eta_out

        self.init_strategy = init_strategy

        self.state = None

        self.device = device

    def reset(self):
        state_p, _ = self.p_data.reset()

        state_p /= self.genbase

        if self.init_strategy == "random":
            res = np.random.dirichlet(np.ones(self.num)) * self.residual
            state_b = res + self.low
        elif self.init_strategy == "empty":
            state_b = np.ones(self.num) * (self.low + 0.1)
        elif self.init_strategy == "full":
            state_b = np.ones(self.num) * self.high
        else:
            raise NotImplementedError

        # print(state_b.shape, state_p.shape)
        self.state = np.concatenate([state_b, state_p])

        return self.state.copy()

    def step(self, action):
        # print("action", action)
        state_p, done = self.p_data.fetch()

        state_p /= self.genbase

        state_b = self.state[:self.num]
        price = self.state[self.num]

        # print(state_p.shape, done)
        # print("price", price)

        up = np.ones(self.num) * self.high
        low = np.ones(self.num) * self.low
        new_p_max = 1 / self.eta_in * np.concatenate((self.p_max * np.ones((self.num, 1)), (up - state_b)[:, None]), axis=1)
        new_p_min = self.eta_out * np.concatenate((self.p_min * np.ones((self.num, 1)), (low - state_b)[:, None]), axis=1)

        new_p_max = np.min(new_p_max, axis=1)
        new_p_min = np.max(new_p_min, axis=1)

        action_new = np.clip(action, new_p_min, new_p_max)
        action_new = self.process_action(action_new)
        state_b += action_new
        self.state = np.concatenate([state_b, state_p])

        cost = - action * price
        cost = cost.sum()

        return cost, self.state.copy()

    def ineq_resid(self, state, action):

        if state.dim() == 1:
            state = state.view(1, -1)

        state = state.cpu().numpy()
        up = np.ones(self.num) * self.high
        low = np.ones(self.num) * self.low
        new_p_max = 1 / self.eta_in * np.concatenate((self.p_max * np.ones((state.shape[0], self.num, 1)), (up - state)[:, :, None]), axis=2)
        new_p_min = self.eta_out * np.concatenate((self.p_min * np.ones((state.shape[0], self.num, 1)), (low - state)[:, :, None]), axis=2)

        new_p_max = torch.tensor(np.min(new_p_max, axis=2), dtype=torch.get_default_dtype(), device=self.device)
        new_p_min = torch.tensor(np.max(new_p_min, axis=2), dtype=torch.get_default_dtype(), device=self.device)
        resids = torch.cat([
            action - new_p_max,
            new_p_min - action,
        ], dim=1)
        return resids

    def ineq_dist(self, state, action):
        resids = self.ineq_resid(state, action)
        return torch.clamp(resids, 0)

    def update_bound(self, state):
        # print(state.shape)
        if state.dim() == 1:
            state = state.view(1, -1)

        state = state.cpu().numpy()
        state = state[:, -self.num-self.p_data.ahead:-self.p_data.ahead]

        up = np.ones(self.num) * self.high
        low = np.ones(self.num) * self.low
        new_p_max = 1 / self.eta_in * np.concatenate((self.p_max * np.ones((state.shape[0], self.num, 1)), (up - state)[:, :, None]), axis=2)
        new_p_min = self.eta_out * np.concatenate((self.p_min * np.ones((state.shape[0], self.num, 1)), (low - state)[:, :, None]), axis=2)

        new_p_min = np.max(new_p_min, axis=2)
        new_p_max = np.min(new_p_max, axis=2)

        return new_p_max, new_p_min

    def process_action(self, action):
        action[action >= 0] *= self.eta_in
        action[action < 0] /= self.eta_out

        return action

class ElectricVehicle(Battery):

    def __init__(self, p_data, a_data, num, low=10, high=80, p_min=-20, p_max=20, eta_in=0.9, eta_out=0.9, init_strategy="full"):

        super(ElectricVehicle, self).__init__(p_data, num, low, high, p_min, p_max, eta_in, eta_out, init_strategy)
        self.a_data = a_data

        self.mask = None

    def step(self, action):

        up = np.ones((self.num, 1))*self.high
        new_high = np.concatenate((self.p_max[:, None], (up - self.state)[:, None]), axis=1)
        new_low = np.concatenate((self.p_min[:, None], (up-self.state)[:, None]), axis=1)

        new_high = np.min(new_high, axis=1)
        new_low = np.max(new_low, axis=1)

        action_new = np.clip(action, new_low, new_high)
        self.state += action_new

        return self.state, self.ineq_dist(action, new_low, new_high)

    def reset(self):
        self.a_data.reset()
        self.p_data.reset()
        if self.init_strategy == "random":
            self.state = np.random.rand(self.num) * (self.high-self.low) + self.low
        elif self.init_strategy == "empty":
            self.state = np.ones(self.num) * self.low
        elif self.init_strategy == "full":
            self.state = np.ones(self.num) * self.high
        else:
            raise NotImplementedError

        self.mask = self.a_data.fetch()

        return self.state.copy(), self.mask.copy()

class EVOPFEnv(gym.Env):
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        ### For Pytorch
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        ## Define useful power network quantities and indices
        ppc = case14()
        self.data = DemandLoader(ppc=ppc, rho=0.5, q_rand=True, T=24, min_pf=0.9, max_pf=1.0, regularized=True)

        self.nahead = 24

        self.p_data = PriceLoader(T=24, ahead=self.nahead, regularized=True, reg_bias=0.1, reg_sigma=0.05)

        self.ppc = ppc

        self.nbus, _ = ppc['bus'].shape

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])
        self.spv_ = np.array([np.where(x == self.spv)[0][0] for x in self.spv])

        self.ng = ppc['gen'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        # evs
        self.ne = self.ng
        self.evs = Battery(self.p_data, num=self.ne, genbase=self.baseMVA, init_strategy="empty", device=self.device)
        self.we = 5.0
        self.wg = 1.0

        self.quad_costs = torch.tensor(ppc['gencost'][:, 4], dtype=torch.get_default_dtype(), device=self.device)
        self.lin_costs = torch.tensor(ppc['gencost'][:, 5], dtype=torch.get_default_dtype(), device=self.device)
        self.const_cost = ppc['gencost'][:, 6].sum()

        self.pmax = torch.tensor(ppc['gen'][:, idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype(), device=self.device)
        self.pmin = torch.tensor(ppc['gen'][:, idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype(), device=self.device)
        self.qmax = torch.tensor(ppc['gen'][:, idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype(), device=self.device)
        self.qmin = torch.tensor(ppc['gen'][:, idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype(), device=self.device)
        self.vmax = torch.tensor(ppc['bus'][:, idx_bus.VMAX], dtype=torch.get_default_dtype(), device=self.device)
        self.vmin = torch.tensor(ppc['bus'][:, idx_bus.VMIN], dtype=torch.get_default_dtype(), device=self.device)
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])],
                                    dtype=torch.get_default_dtype(), device=self.device).squeeze(-1)

        ppc2 = deepcopy(ppc)
        ppc2['bus'][:, 0] -= 1
        ppc2['branch'][:, [0, 1]] -= 1
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype(), device=self.device)
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype(), device=self.device)

        ## Define optimization problem input and output variables

        self._xdim = 2 * self.nbus
        self._ydim = 2 * self.ng + 2 * self.nbus + self.ne

        self._neq = 2 * self.nbus
        self._nineq = 4 * self.ng + 2 * self.nbus
        self._nknowns = self.nslack

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2 * self.ng
        self.va_start_yidx = 2 * self.ng + self.nbus
        self.pe_start_yidx = 2 * self.ng + 2 * self.nbus

        ## Define variables and indices for "partial completion" neural network

        # pg (non-slack) and |v|_g (including slack)
        self._partial_vars = np.concatenate([self.pg_start_yidx + self.pv_,
                                             self.vm_start_yidx + self.spv,
                                             self.va_start_yidx + self.slack, self.pe_start_yidx + self.spv_])
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])
        self._partial_actions = np.concatenate(
            [self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv
                , self.pe_start_yidx + self.spv_])
        self._other_actions = np.setdiff1d(np.arange(self.ydim), self._partial_vars)


        # initial values for solver
        self.vm_init = ppc['bus'][:, idx_bus.VM]
        self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])
        self.pg_init = ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = ppc['gen'][:, idx_gen.QG] / self.genbase

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2 * self.npv + self.nslack)
        self.pe_pv_zidx = np.arange(2 * self.npv + self.nslack + 1, 2 * self.npv + self.nslack + self.ne)
        self.pe_spv_zidx = np.arange(2 * self.npv + self.nslack, 2 * self.npv + self.nslack + self.ne)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus

        ### For gym_env

        pd_max = 10.0
        qd_max = 10.0
        price_max = 240.0
        price_min = 0.0
        high = np.array([pd_max]*self.nbus + [qd_max]*self.nbus + [self.evs.high] * self.ne + [price_max] * self.nahead)
        low = np.array([-pd_max]*self.nbus + [-qd_max]*self.nbus + [self.evs.low] * self.ne + [price_min] * self.nahead)
        action_high = np.concatenate([np.array(torch.cat((self.pmax, self.qmax, self.vmax)).cpu()),
                                      np.array([np.pi]*self.nbus), np.array([self.evs.p_max]*self.ne)])
        action_low = np.concatenate([np.array(torch.cat((self.pmin, self.qmin, self.vmin)).cpu()),
                                      np.array([-np.pi]*self.nbus), np.array([self.evs.p_min]*self.ne)])
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.eq_num = 28
        self.ineq_num = 58

        self.state = None
        self.action = None
        self.state_grid = None
        self.state_evs = None

        self.viewer = None

        self.volatile = True

    def step(self, action):

        reward_grid = - self.obj_fn(action).cpu().numpy()
        reward_evs, next_state_evs = self.evs.step(action[self.pe_start_yidx:])
        # print("grid:", reward_grid, "evs:", reward_evs)
        reward = self.wg * reward_grid + self.we * reward_evs
        next_state_grid, done = self.data.fetch()

        next_state = np.concatenate([next_state_grid, next_state_evs])
        state = self.state
        self.state = next_state

        #render
        self.action = action
        grid, evs = self.decompose(state)
        self.state_grid = grid.copy()
        self.state_evs = evs.copy()

        return state, reward, done, {'ineq_viol': self.ineq_dist_np(state, action), 'eq_viol': self.eq_resid_np(state, action)}


    def reset(self):
        state_grid, _ = self.data.reset()
        state_evs = self.evs.reset()

        self.state = np.concatenate([state_grid, state_evs])

        # render
        self.action = None
        self.state_grid = None
        self.state_evs = None

        return self.state.copy()

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 600
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if self.action is None or self.state_grid is None or self.state_evs is None:
            raise NotImplementedError("cannot implement render before step!")
        graph = igraph.Graph()
        pv_color = 'orange'
        num_bus = len(self.ppc['bus'])
        num_gen = len(self.ppc['gen'])
        generator_list = [int(gen_info[0]) for gen_info in self.ppc['gen']]
        # print(f"generator_list: {generator_list}")
        # generator_list: [1, 2, 3, 6, 8]
        # print(f"self.pd_max: {self.pd_max}")
        # print(f"self.state: {self.state}")
        s = 20
        label_size = 15
        for bus_idx in range(num_bus):
            # enumerate all the bus
            shape = 'circle'
            if bus_idx + 1 in generator_list:
                # the bus is a generator
                if self.action is not None:
                    gen_idx = generator_list.index(bus_idx + 1)
                    color_ratio = self.action[gen_idx] / self.pmax[gen_idx]
                    c = (1, 1 - color_ratio, 1 - color_ratio)
                else:
                    c = (1, 1, 1)
            else:
                c = (1, 1 - 0.5 * self.state_grid[bus_idx], 1 - self.state_grid[bus_idx])
            graph.add_vertex(color=c, size=s, label=f"{bus_idx + 1}", shape=shape, label_dist=2,
                             label_size=label_size, )

        for edge_info in self.ppc['branch']:
            graph.add_edge(int(edge_info[0]) - 1, int(edge_info[1]) - 1)

        for gen_idx, generator in enumerate(generator_list):
            shape = 'square'
            if self.state_evs is not None:
                color_ratio = (self.state_evs[gen_idx] - self.evs.low) / self.evs.high
                c = (1 - color_ratio, 1 - color_ratio, 1)
            else:
                c = (1, 1, 1)
            graph.add_vertex(color=c, size=s, label=f"{generator}", shape=shape, label_dist=2,
                             label_size=label_size, )
            if self.action is not None:
                color_ratio = self.action[gen_idx - num_gen] / self.evs.p_max
                if color_ratio > 0:
                    c = (1, 1 - color_ratio, 1 - color_ratio)
                else:
                    c = (1 + color_ratio, 1, 1 + color_ratio)
            graph.add_edge(generator - 1, gen_idx + num_bus, color=c)

        igraph.plot(graph, "tmp.png", margin=50, curved=True)
        imgtrans = rendering.Transform(translation=(300, 300))
        img = rendering.Image("tmp.png", 600, 600)
        img.add_attr(imgtrans)
        self.viewer.add_onetime(img)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

        if os.path.exists("tmp.png"):
            os.remove("tmp.png")

    def decompose(self, state):
        return state[:2*self.nbus], state[2*self.nbus:]

    @property
    def partial_actions(self):
        return self._partial_actions

    @property
    def other_actions(self):
        return self._other_actions

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_unknown_vars

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def device(self):
        return self._device

    def get_action_vars(self, action):
        pg = action[:, :self.ng]
        qg = action[:, self.ng:2 * self.ng]
        vm = action[:, 2 * self.ng:2 * self.ng + self.nbus]
        va = action[:, -self.ne-self.nbus:-self.ne]
        pe = action[:, -self.ne:]
        return pg, qg, vm, va, pe

    def obj_fn(self, action):
        action = torch.tensor(action, device=self.device)
        if action.dim() == 1:
            action = action.view(1,-1)
        pg, _, _, _, _ = self.get_action_vars(action)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw ** 2).sum(axis=1) + \
               (self.lin_costs * pg_mw).sum(axis=1) + \
               self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, state, action):
        pg, qg, vm, va, pe = self.get_action_vars(action)

        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)

        ## power balance equations
        tmp1 = vr @ self.Ybusr - vi @ self.Ybusi
        tmp2 = -vr @ self.Ybusi - vi @ self.Ybusr

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg + pe
        real_resid = (pg_expand - state[:, :self.nbus]) - (vr * tmp1 - vi * tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg
        react_resid = (qg_expand - state[:, self.nbus:2*self.nbus]) - (vr * tmp2 + vi * tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)

        return resids

    def ineq_resid(self, state, action):
        pg, qg, vm, va, pe = self.get_action_vars(action)
        resids_grid = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm
        ], dim=1)

        resids_evs = self.evs.ineq_resid(state[:, 2*self.nbus:2*self.nbus+self.ne], pe)

        resids = torch.cat([resids_grid, resids_evs], dim=1)

        return resids

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
        eq_jac = self.eq_jac(action)
        eq_resid = self.eq_resid(state, action)
        return 2 * eq_jac.transpose(1, 2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, state, action, eps=0.0):
        ineq_jac = self.ineq_jac(state, action)
        ineq_dist = self.ineq_dist(state, action)
        ineq_dist = ineq_dist + (ineq_dist > 0) * eps
        return 2 * ineq_jac.transpose(1, 2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    def ineq_grad_new(self, state, action, eps=0.0):
        ineq_jac = self.ineq_jac(state, action)
        ineq_dist = self.ineq_dist(state, action)
        ineq_dist = (ineq_dist > 0).to(torch.float32)
        return ineq_jac.transpose(1, 2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    def ineq_partial_grad(self, state, action, eps=0.0):
        eq_jac = self.eq_jac(action)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])

        # direct_grad = self.ineq_grad(state, action, eps)

        direct_grad = self.ineq_grad_new(state, action, eps)
        indirect_partial_grad = dynz_dz.transpose(1, 2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(state.shape[0], self.ydim, device=self.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

        return full_grad

    def eq_jac(self, action):
        _, _, vm, va, _ = self.get_action_vars(action)

        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi
        YrvrYivi = vr @ Yr - vi @ Yi
        YivrYrvi = vr @ Yi + vi @ Yr

        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva)) \
                    - mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi) - Ydiagv(Yi, vr)) \
                    - mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi) + Ydiagv(Yr, vr))
        dreal_dpe = torch.zeros(self.nbus, self.ng, device=self.device)
        dreal_dpe[self.spv, :] = - torch.eye(self.ng, device=self.device)

        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva)) \
                     - mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi) + Ydiagv(Yr, vr)) \
                     - mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi) - Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape),
                       torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device),
                       dreal_dvm, dreal_dva,
                       dreal_dpe.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape)], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device),
                       dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                       dreact_dvm, dreact_dva,
                       torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device)], dim=2)],
            dim=1)

        return jac

    def ineq_jac(self, state, action):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                       torch.eye(self.ng, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                       -torch.eye(self.ng, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device),
                       torch.zeros(self.nbus, self.ng, device=self.device),
                       torch.eye(self.nbus, device=self.device),
                       torch.zeros(self.nbus, self.nbus, device=self.device),
                       torch.zeros(self.nbus, self.ng, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device),
                       torch.zeros(self.nbus, self.ng, device=self.device),
                       -torch.eye(self.nbus, device=self.device),
                       torch.zeros(self.nbus, self.nbus, device=self.device),
                       torch.zeros(self.nbus, self.ng, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.eye(self.ng, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                       torch.zeros(self.ng, self.ng, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       torch.zeros(self.ng, self.nbus, device=self.device),
                       -torch.eye(self.ng, device=self.device)], dim=1),
        ], dim=0)

        return jac.unsqueeze(0).expand(action.shape[0], *jac.shape)

    # Processes intermediate neural network output
    def process_output(self, X, out):
        out2 = nn.Sigmoid()(out[:, :-self.nbus + self.nslack])
        pg = out2[:, :self.qg_start_yidx] * self.pmax + (1 - out2[:, :self.qg_start_yidx]) * self.pmin
        qg = out2[:, self.qg_start_yidx:self.vm_start_yidx] * self.qmax + \
             (1 - out2[:, self.qg_start_yidx:self.vm_start_yidx]) * self.qmin
        vm = out2[:, self.vm_start_yidx:] * self.vmax + (1 - out2[:, self.vm_start_yidx:]) * self.vmin

        va = torch.zeros(X.shape[0], self.nbus, device=self.device)
        va[:, self.nonslack_idxes] = out[:, self.va_start_yidx:]
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device).unsqueeze(0).expand(X.shape[0], self.nslack)

        return torch.cat([pg, qg, vm, va], dim=1)

    # Solves for the full set of variables
    def complete_partial(self, state, action_partial):
        if action_partial.dim() == 1:
            action_partial = action_partial.view(1, -1)
        return PFFunction(self)(state[:, :self.xdim], action_partial)

    def opt_solve(self, X, solver_type='pypower', tol=1e-4):
        X_np = X.detach().cpu().numpy()

        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:, idx_bus.VMIN] = ppc['bus'][:, idx_bus.VMIN]
        ppc['bus'][:, idx_bus.VMAX] = ppc['bus'][:, idx_bus.VMAX]

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = opf(ppc, ppopt)
            end_time = time.time()
            total_time += (end_time - start_time)

            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time / len(X_np)

    @property
    def box_constraint(self):
        return self.action_space.low, self.action_space.high

    @property
    def box_constraint_partial(self):
        return self.action_space.low[self.partial_actions], self.action_space.high[self.partial_actions]

    def update(self, state, full=False):
        n = state.shape[0]
        new_p_max, new_p_min = self.evs.update_bound(state)

        if full:
            low, high = self.box_constraint
        else:
            low, high = self.box_constraint_partial

        low = low[None].repeat(n, axis=0)
        high = high[None].repeat(n, axis=0)
        low[:, -self.ne:] = new_p_min
        high[:, -self.ne:] = new_p_max

        return low, high


def PFFunction(env, tol=1e-5, bsz=256, max_iters=50):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, state, action_partial):

            DEVICE = env.device

            ## Step 1: Newton's method
            action = torch.zeros(state.shape[0], env.ydim, device=DEVICE)
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            action[:, env.pg_start_yidx + env.pv_] = action_partial[:, env.pg_pv_zidx]  # pg at non-slack gens
            action[:, env.vm_start_yidx + env.spv] = action_partial[:, env.vm_spv_zidx]  # vm at gens
            action[:, env.va_start_yidx + env.slack] = torch.tensor(env.slack_va, dtype=torch.get_default_dtype(), device=DEVICE)  # va at slack bus
            action[:, env.pe_start_yidx + env.spv_] = action_partial[:, env.pe_spv_zidx]

            # init guesses for remaining values
            action[:, env.vm_start_yidx + env.pq] = torch.tensor(env.vm_init[env.pq], dtype=torch.get_default_dtype(), device=DEVICE)  # vm at load buses
            action[:, env.va_start_yidx + env.pv] = torch.tensor(env.va_init[env.pv], dtype=torch.get_default_dtype(),
                                                            device=DEVICE)  # va at non-slack gens
            action[:, env.va_start_yidx + env.pq] = torch.tensor(env.va_init[env.pq], dtype=torch.get_default_dtype(), device=DEVICE)  # va at load buses
            action[:, env.qg_start_yidx:env.qg_start_yidx + env.ng] = 0  # qg at gens (not used in Newton upd)
            action[:, env.pg_start_yidx + env.slack_] = 0  # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                env.pflow_start_eqidx + env.pv,  # real power flow at non-slack gens
                env.pflow_start_eqidx + env.pq,  # real power flow at load buses
                env.qflow_start_eqidx + env.pq])  # reactive power flow at load buses
            newton_guess_inds = np.concatenate([
                env.vm_start_yidx + env.pq,  # vm at load buses
                env.va_start_yidx + env.pv,  # va at non-slack gens
                env.va_start_yidx + env.pq])  # va at load buses

            converged = torch.zeros(state.shape[0])
            jacs = []
            newton_jacs_inv = []
            for b in range(0, state.shape[0], bsz):
                # print('batch: {}'.format(b))
                state_b = state[b:b + bsz]
                action_b = action[b:b + bsz]

                for i in range(max_iters):
                    # print(i)
                    gy = env.eq_resid(state_b, action_b)[:, keep_constr]
                    jac_full = env.eq_jac(action_b)
                    jac = jac_full[:, keep_constr, :]
                    newton_jac_inv = torch.inverse(jac[:, :, newton_guess_inds])
                    delta = newton_jac_inv.bmm(gy.unsqueeze(-1)).squeeze(-1)
                    action_b[:, newton_guess_inds] -= delta
                    if torch.norm(delta, dim=1).abs().max() < tol:
                        break

                converged[b:b + bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                newton_jacs_inv.append(newton_jac_inv)

            ## Step 2: Solve for remaining variables

            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            action[:, env.qg_start_yidx:env.qg_start_yidx + env.ng] = \
                -env.eq_resid(state, action)[:, env.qflow_start_eqidx + env.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            action[:, env.pg_start_yidx + env.slack_] = \
                -env.eq_resid(state, action)[:, env.pflow_start_eqidx + env.slack]

            ctx.env = env
            ctx.save_for_backward(torch.cat(jacs), torch.cat(newton_jacs_inv),
                                  torch.tensor(newton_guess_inds, device=DEVICE),
                                  torch.tensor(keep_constr, device=DEVICE))

            return action

        @staticmethod
        def backward(ctx, dl_dy):

            env = ctx.env
            jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors

            DEVICE = env.device

            ## Step 2 (calc pg at slack and qg at gens)

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([env.pflow_start_eqidx + env.slack, env.qflow_start_eqidx + env.spv])
            last_vars = np.concatenate([
                env.pg_start_yidx + env.slack_, np.arange(env.qg_start_yidx, env.qg_start_yidx + env.ng)])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, env.vm_start_yidx:env.pe_start_yidx].transpose(1, 2).bmm(
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars]

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=DEVICE)
            dl_dy_3[:, env.vm_start_yidx:env.pe_start_yidx] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], env.xdim, device=DEVICE)
            dl_dx_3[:, np.concatenate([env.slack, env.nbus + env.spv])] = dl_dpdqd_3

            ## Step 1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            d_int = newton_jac_inv.transpose(1, 2).bmm(
                dl_dy_total[:, newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], env.npv + env.ng + env.ne, device=DEVICE)
            dl_dz_2[:, env.pg_pv_zidx] = -d_int[:, :env.npv]  # dl_dpg at pv buses
            dl_dz_2[:, env.vm_spv_zidx] = -jac2[:, :, env.vm_start_yidx + env.spv].transpose(1, 2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)
            dl_dz_2[:, env.pe_pv_zidx] = d_int[:, :env.npv]
            dl_dz_2[:, 2 * env.npv + env.nslack + env.slack] = dl_dx_3[:, env.slack]

            dl_dx_2 = torch.zeros(dl_dy.shape[0], env.xdim, device=DEVICE)
            dl_dx_2[:, env.pv] = d_int[:, :env.npv]  # dl_dpd at pv buses
            dl_dx_2[:, env.pq] = d_int[:, env.npv:env.npv + len(env.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, env.nbus + env.pq] = d_int[:, -len(env.pq):]  # dl_dqd at pq buses

            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                env.pg_start_yidx + env.pv_, env.vm_start_yidx + env.spv, env.pe_start_yidx + env.spv_])]

            return dl_dx_total, dl_dz_total

    return PFFunctionFn.apply
