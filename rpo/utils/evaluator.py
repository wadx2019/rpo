# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Evaluator."""
import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms."""

    def evaluate(
        self,
        env,
        actor,
        device,
        t,
        num_episodes: int = 10,
        cost_criteria: float = 1.0
    ):
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int): number of episodes to evaluate the agent.
            cost_criteria (float): the cost criteria for the evaluation.

        Returns:
            (float, float, float): the average return, the average cost, and the average length of the episodes.
        """
        self._env = copy.deepcopy(env)
        self._actor = actor
        if self._env is None or self._actor is None:
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.'
            )

        total_rewards = []
        mean_ineq_viols = []
        mean_eq_viols = []
        max_ineq_viols = []
        max_eq_viols = []
        length = 0

        for episode in range(num_episodes):
            total_reward = 0
            mean_ineq_viol = 0
            mean_eq_viol = 0
            max_ineq_viol = 0
            max_eq_viol = 0
            length = 0
            obs, _ = self._env.reset()

            done = False
            while not done:
                with torch.no_grad():
                    act = self._actor.predict(
                        torch.as_tensor(obs, dtype=torch.float32).to(device),
                        deterministic=True,
                    )
                    act = act.cpu()
                self._env.update(obs)
                obs, rew, cost, terminated, truncated, info = self._env.step(act)

                total_reward += rew.item()
                mean_ineq_viol += (info["ineq_viol"].max()-mean_ineq_viol) / (length + 1)
                mean_eq_viol += (np.abs(info["eq_viol"]).max() - mean_eq_viol) / (length + 1)
                max_ineq_viol = max(max_ineq_viol, info["ineq_viol"].max())
                max_eq_viol = max(max_eq_viol, np.abs(info["eq_viol"]).max())

                done = bool(terminated or truncated)
                length += 1

            total_rewards.append(total_reward)
            mean_ineq_viols.append(mean_ineq_viol)
            mean_eq_viols.append(mean_eq_viol)
            max_ineq_viols.append(max_ineq_viol)
            max_eq_viols.append(max_eq_viol)

        total_rewards = np.array(total_rewards)
        mean_ineq_viols = np.array(mean_ineq_viols)
        mean_eq_viols = np.array(mean_eq_viols)
        max_ineq_viols = np.array(max_ineq_viols)
        max_eq_viols = np.array(max_eq_viols)

        rmean, rstd, ineqmean, ineqstd, eqmean, eqstd, maxineqmean, maxineqstd, \
        maxeqmean, maxeqstd = total_rewards.mean(), total_rewards.std(), mean_ineq_viols.mean(), mean_ineq_viols.std(), \
        mean_eq_viols.mean(), mean_eq_viols.std(), max_ineq_viols.mean(), max_ineq_viols.std(), \
        max_eq_viols.mean(), max_eq_viols.std()

        print(f"Eval: epoch {t}, rewards: {rmean:.4f}({rstd:.4f}), mean_ineq_viol: {ineqmean:.4f}({ineqstd:.4f})," +
              f" mean_eq_viol: {eqmean:.4f}({eqstd:.4f}), max_ineq_viol: {maxineqmean:.4f}({maxineqstd:.4f})" +
              f" max_eq_viol: {maxeqmean:.4f}({maxeqstd:.4f})")


