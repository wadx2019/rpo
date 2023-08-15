import datetime
import numpy as np
import pickle

class Logger(object):

    def __init__(self, keys, epochs=1000, times=3, name="twddpg"):
        self.tracker = {key: np.zeros((epochs*times,)) for key in keys}
        self.name = name
        self.pointer = 0
        self.epochs = epochs
        self.times = times

    def add(self, **kwargs):
        if self.pointer >= self.epochs * self.times:
            raise StopIteration("logger is full!")

        for key in kwargs.keys():
            self.tracker[key][self.pointer] = kwargs[key]

        self.pointer += 1

    def save(self, path):
        with open(path+"_"+datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"), "wb") as f:
            pickle.dump(self, f)