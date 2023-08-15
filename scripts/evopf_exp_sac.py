import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from rpo.algo import RPOSAC
from rpo.env import *
from rpo.utils.logger import Logger
import gym
import torch
from rpo.utils.monitor import get_monitor
mon = get_monitor(None, filepath=os.path.abspath(__file__))
import time

np.random.seed(123)
torch.manual_seed(123)

torch.set_default_dtype(torch.float32)

def main():
    max_epochs = 40000
    l = []
    name = "evopf_sac_partial"

    times = 3

    logger = Logger(("epoch", "reward", "max_ineq", "max_eq"), times=times, epochs=max_epochs, name=name)

    for _ in range(times):

        env = gym.make("EVOPF-v0")

        twagent = RPOSAC(env, "./test", name=name, logger=logger, batch_size=256, max_steps=10, warmup=0, lr_dual=2e-2, corr_lr=1e-4,  eps=0.0001, eps_start=0.0001, lr_actor=1e-4, lr_critic=3e-4,
                         eps_epoch=20000, eval_lr=1e-4, eval_steps=50, grad_eps=0.1, corr_momentum=0.0, policy_fre=4, ex_action_dim=1, max_epochs=max_epochs,
                         fixed=False, init_lamb=0.0, init_nju=0.0, alpha=0.001, automatic_entropy_tuning=False,
                         capacity=20000, clip_thres=0.2, shared_param=False, value_type="cat", partial=True, partial_idx=np.random.choice(np.arange(env.action_dim), size=env.action_dim-env.eq_num-1, replace=False)
                         )
        start = time.time()
        twagent.run(eval=False)
        l.append(time.time() - start)
        env.close()
    print(sum(l) / times)
    logger.save(os.path.join("./test", name))

if __name__ == "__main__":
    main()
