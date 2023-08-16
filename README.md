# RPO
Official implementation of Reduced Policy Optimization

The code for the submission of NeurIPS2023, ID 2211 :"Reduced Policy Optimization for Continuous Control with Hard Constraints".

To run the experiments, you need to first install the python package rpo via running `pip install .` in the current directory.

Then, You can simply 

run `python scripts/cart_exp.py` to re-implement our experiments on the Safe Cartpole environment using RPODDPG algorithm.

run `python scripts/pen_exp.py` to re-implement our experiments on the Spring Pendulum environment using RPODDPG algorithm.

run `python scripts/evopf_exp.py` to re-implement our experiments on the OPF with Battery Energy Storage environment using RPODDPG algorithm.

run `python scripts/cart_exp_sac.py` to re-implement our experiments on the Safe Cartpole environment using RPOSAC algorithm.

run `python scripts/pen_exp_sac.py` to re-implement our experiments on the Spring Pendulum environment using RPOSAC algorithm.

run `python scripts/evopf_exp_sac.py` to re-implement our experiments on the OPF with Battery Energy Storage environment using RPOSAC algorithm.

