import numpy as np
from rpo.algo import RPODDPG
from rpo.env import *
from rpo.utils.logger import Logger
import gym
import torch
import os
import time

np.random.seed(111)
torch.manual_seed(123)

torch.set_default_dtype(torch.float32)

def main():
    l = []

    max_epochs = 40000

    name = "evopf_ddpg"

    times = 3

    logger = Logger(("epoch", "reward", "max_ineq", "max_eq"), times=times, epochs=max_epochs, name=name)

    for _ in range(times):

        env = gym.make("EVOPF-v0")

        twagent = RPODDPG(env, "./test", name=name, logger=logger, batch_size=256, max_steps=10, warmup=0, lr_dual=2e-2, corr_lr=1e-4, eps=0.0001, eps_start=0.0001,
                         eps_epoch=20000, eval_lr=1e-4, eval_steps=50, grad_eps=0.02, corr_momentum=0.0, policy_fre=4, ex_action_dim=1, gamma=0.95, max_epochs=max_epochs,
                         capacity=20000, clip_thres=0.2, shared_param=False, value_type="cat")
        start = time.time()
        twagent.run()
        l.append(time.time() - start)
        env.close()

    print(sum(l)/times)
    logger.save(os.path.join("./test", name))

if __name__ == "__main__":
    main()