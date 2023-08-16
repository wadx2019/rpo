import numpy as np
from rpo.algo import RPOSAC
from rpo.env import *
from rpo.utils.logger import Logger
import gym
import torch
import os


np.random.seed(123)
torch.manual_seed(123)

def main():
    max_epochs = 20000

    name = "cart_sac"

    times = 5

    logger = Logger(("epoch", "reward", "max_ineq", "max_eq"), times=times, epochs=max_epochs, name=name)

    for _ in range(times):

        env = gym.make("CartSafe-v0")

        rpoagent = RPOSAC(env, "./test", name=name, logger=logger, batch_size=256, max_steps=10, warmup=0, lr_dual=0.2, corr_lr=2e-2, eps=5e-3, eps_start=5e-3, lr_actor=1e-4, lr_critic=3e-4,
                         eps_epoch=20000, eval_lr=2e-2, eval_steps=50, grad_eps=0.1, corr_momentum=0.0, policy_fre=4, max_epochs=max_epochs,
                         alpha=0.1, automatic_entropy_tuning=False,
                         capacity=20000, shared_param=False, value_type="add", clip_thres=0.2, embed_dim=128, hidden_dim=256)
        rpoagent.run()

        env.close()

    logger.save(os.path.join("./test", name))

if __name__ == "__main__":
    main()
