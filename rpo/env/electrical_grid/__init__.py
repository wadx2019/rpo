from .evopf import EVOPFEnv
from gym.envs.registration import register

register(
    id="EVOPF-v0",
    entry_point="rpo.env:EVOPFEnv",
)


