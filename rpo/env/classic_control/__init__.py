from .cartpole import CartSafeEnv
from .pendulum import SpringPendulumEnv

from gym.envs.registration import register

register(
    id="CartSafe-v0",
    entry_point="rpo.env:CartSafeEnv",
    max_episode_steps=200,
)

register(
    id="SpringPendulum-v0",
    entry_point="rpo.env:SpringPendulumEnv",
    max_episode_steps=200,
)