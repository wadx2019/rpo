import numpy as np

class Agent(object):

    def __init__(self, state_dim, action_dim, box_constraint):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.box_constraint = box_constraint


    def take_action(self, state, deterministic=False):

        action = np.random.rand(self.action_dim)
        action = self.box_constraint(action)

        return action

