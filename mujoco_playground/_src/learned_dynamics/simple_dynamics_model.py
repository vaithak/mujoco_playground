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
"""Simple MLP-based dynamics model for learning environment transitions."""

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jp


class SimpleDynamicsModel(nn.Module):
  """Simple MLP-based dynamics model for learning environment transitions.

  Predicts next state (qpos, qvel) given current state and action.
  """

  hidden_dims: Tuple[int, ...] = (256, 256)

  @nn.compact
  def __call__(
      self, qpos: jax.Array, qvel: jax.Array, action: jax.Array
  ) -> Tuple[jax.Array, jax.Array]:
    """Predict next state given current state and action.

    Args:
        qpos: Current joint positions
        qvel: Current joint velocities
        action: Action taken

    Returns:
        Tuple of (next_qpos, next_qvel)
    """
    # Concatenate inputs
    x = jp.concatenate([qpos, qvel, action], axis=-1)

    # MLP layers
    for dim in self.hidden_dims:
      x = nn.Dense(dim)(x)
      x = nn.relu(x)

    # Output next state (predict delta for stability)
    qpos_dim = qpos.shape[-1]
    qvel_dim = qvel.shape[-1]

    delta_qpos = nn.Dense(qpos_dim)(x)
    delta_qvel = nn.Dense(qvel_dim)(x)

    # Add deltas to current state for stability
    next_qpos = qpos + delta_qpos
    next_qvel = qvel + delta_qvel

    return next_qpos, next_qvel


def create_dynamics_model_fn(params):
  """Create a callable dynamics model from trained parameters.

  Args:
      params: Trained model parameters

  Returns:
      A function that takes (state_data, action) and returns updated state_data
  """
  model = SimpleDynamicsModel()

  def dynamics_fn(state_data, action):
    """Predict next state given current state and action.

    Args:
        state_data: Current MJX Data object
        action: Action to take

    Returns:
        Updated state_data with predicted next state
    """
    next_qpos, next_qvel = model.apply(
        params, state_data.qpos, state_data.qvel, action
    )
    # Return updated state_data with predicted next state
    return state_data.replace(qpos=next_qpos, qvel=next_qvel)

  return dynamics_fn
