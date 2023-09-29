import numpy as np
import pickle
import pypower.idx_gen as idx_gen
import pypower.idx_bus as idx_bus
import os

class PriceLoader(object):

    def __init__(self, T=24, ahead=24, regularized=True, reg_bias=0.1, reg_sigma=0.05):
        with open(os.path.dirname(__file__) + "/crawlers/price.pickle", 'rb') as f:
            self.data = np.array(pickle.load(f)["SYS"])

        self.total = len(self.data) // T
        self.curve = np.zeros(T)
        for i in range(self.total):
            self.curve += self.data[i*T:i*T+T]
        self.curve /= self.total
        self.T = T
        self.ahead = ahead

        self.regularized = regularized
        self.reg_bias = reg_bias
        self.reg_sigma = reg_sigma

        self.cache = None
        self.counter = None


    def reset(self):
        self.counter = 0
        if self.regularized:
            tmp = np.concatenate([self.curve, np.zeros(self.T)])
        else:
            dy = np.random.randint(self.total)
            tmp = self.data["SYS"][dy*self.T: dy*self.T+self.T+self.ahead]
        self.cache = self.process(tmp)

        return self.fetch()

    def process(self, pc):
        if self.regularized:
            mag_r = (self.reg_bias * np.random.randn() + 1)
            price = mag_r * pc * (1+np.random.randn(self.T*2))
        else:
            price = pc

        return price

    def fetch(self):

        done = self.counter >= self.T
        tmp = np.zeros(self.ahead) if done else self.cache[self.counter:self.counter+self.ahead].copy()
        self.counter += 1

        return tmp, done

