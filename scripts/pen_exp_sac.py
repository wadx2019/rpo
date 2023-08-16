import numpy as np
import os
from rpo.algo import RPOSAC
from rpo.env import *
from rpo.utils.logger import Logger
import gym
import torch

np.random.seed(123)
torch.manual_seed(121)

def main():

    max_epochs = 20000

    name = "pen_sac"

    times = 5

    logger = Logger(("epoch", "reward", "max_ineq", "max_eq"), times=times, epochs=max_epochs, name=name)

    for _ in range(times):

        env = gym.make("SpringPendulum-v0")

        rpoagent = RPOSAC(env, "./test", name=name, logger=logger, batch_size=256, max_steps=10, warmup=0, lr_dual=0.01, corr_lr=2e-3, eps=1e-2, eps_start=1e-2, lr_actor=1e-4, lr_critic=3e-4,
                         eps_epoch=20000, eval_lr=2e-3, eval_steps=50, grad_eps=0.1, corr_momentum=0.0, policy_fre=4, max_epochs=max_epochs,
                         alpha=0.01, automatic_entropy_tuning=False,
                         capacity=20000, shared_param=False, value_type="add", clip_thres=0.2, embed_dim=128, hidden_dim=256)

        rpoagent.run()
        env.close()

    logger.save(os.path.join("./test", name))

if __name__ == "__main__":
    main()
