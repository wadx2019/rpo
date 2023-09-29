import numpy as np
import pickle
import pypower.idx_gen as idx_gen
import pypower.idx_bus as idx_bus
import os

class ArrivalLoader(object):

    def __init__(self, ppc, rho=0.5, q_rand=False, T=24, min_pf=0.9, max_pf=1.0, regularized=True):
        with open(os.path.dirname(__file__) + "/crawlers/demand.pickle", 'rb') as f:
            self.data = np.array(pickle.load(f)["value"])

        self.total = len(self.data) // T
        self.curve = np.zeros(T)
        for i in range(self.total):
            self.curve += self.data[i*T: i*T+T]
        self.curve /= self.total
        self.T = T
        self.q_rand = q_rand
        self.rho = rho

        self.baseMVA = ppc["baseMVA"]
        self.bus = ppc["bus"]
        self.gen = ppc["gen"]

        self.min_pf = min_pf
        self.max_pf = max_pf

        self.regularized = regularized

        self.cache = None
        self.counter = None


    def reset(self):
        self.counter = 0
        if self.regularized:
            tmp = self.curve
        else:
            dy = np.random.randint(self.total)
            tmp = self.data["value"][dy*self.T: dy*self.T+self.T]
        pd, qd = self.process(tmp)
        self.cache = np.concatenate((pd, qd), axis=1)

        return self.fetch()

    def process(self, ps):
        nbus, _ = self.bus.shape
        idx = np.nonzero(self.bus[idx_bus.PD])
        nnz = len(idx)
        p_total = self.bus[:, idx_bus.PD].sum() * self.T
        ps = ps / ps.sum() * p_total    # (T,)
        ps = ps[:, None]
        rand_mat = np.random.dirichlet(np.ones(nnz), size=self.T) # (T, nnz)
        p_ratio = self.bus[:, idx_bus.PD] / self.bus[:, idx_bus.PD].sum()

        p_ratio = p_ratio[None].repeat(self.T, axis=0) * (1-self.rho)
        p_ratio[:, idx] += self.rho * rand_mat # (T, nbus)

        pd = p_ratio * ps

        if self.q_rand:
            PFFactor = np.random.rand((self.T, nbus)) * (self.max_pf-self.min_pf) + self.min_pf
            qd = pd * np.tan(np.arccos(PFFactor)) * np.sign(self.bus[:, idx_bus.QD])
        else:
            qd = self.bus[:, idx_bus.QD]
        return pd, qd

    def fetch(self):

        done = self.counter >= self.T
        tmp = None if done else self.cache[self.counter].copy()
        self.counter += 1

        return tmp, done

if __name__=="__main__":
    from pypower.api import case14
    ppc = case14()
    ex = ArrivalLoader(ppc=ppc)
    import matplotlib.pyplot as plt
    plt.plot(ex.curve)
    plt.show()