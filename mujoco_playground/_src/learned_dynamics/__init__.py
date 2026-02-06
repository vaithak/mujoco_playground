# Copyright 2025 DeepMind Technologies Limited
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
"""Learned dynamics module for model-based RL."""

from mujoco_playground._src.learned_dynamics.learned_env_wrapper import LearnedDynamicsEnv
from mujoco_playground._src.learned_dynamics.simple_dynamics_model import SimpleDynamicsModel
from mujoco_playground._src.learned_dynamics.simple_dynamics_model import create_dynamics_model_fn
from mujoco_playground._src.learned_dynamics.train_dynamics_model import collect_rollout_data
from mujoco_playground._src.learned_dynamics.train_dynamics_model import train_dynamics_model

__all__ = [
    'LearnedDynamicsEnv',
    'SimpleDynamicsModel',
    'create_dynamics_model_fn',
    'train_dynamics_model',
    'collect_rollout_data',
]
