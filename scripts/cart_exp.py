import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from rpo.algo import RPODDPG
from rpo.env import *
from rpo.utils.logger import Logger
import gym
import torch
import time

np.random.seed(123)
torch.manual_seed(123)

def main():
    l= []

    max_epochs = 20000

    name = "cart_ddpg"

    times = 5

    logger = Logger(("epoch", "reward", "max_ineq", "max_eq"), times=times, epochs=max_epochs, name=name)

    for _ in range(times):

        env = gym.make("CartSafe-v0")

        twagent = RPODDPG(env, "./test", name=name, logger=logger, batch_size=256, max_steps=10, warmup=0, lr_dual=0.2, corr_lr=2e-2, eps=1.0, eps_start=1.0, lr_actor=1e-4, lr_critic=3e-4,
                         eps_epoch=20000, eval_lr=2e-2, eval_steps=50, grad_eps=0.1, corr_momentum=0.0, policy_fre=4, max_epochs=max_epochs,
                         capacity=20000, shared_param=True, value_type="add", clip_thres=0.2, embed_dim=128, hidden_dim=256)

        start = time.time()
        twagent.run(eval=False)
        l.append(time.time() - start)
        env.close()

    print(sum(l) / times)
    logger.save(os.path.join("./test", name))

if __name__ == "__main__":
    main()
