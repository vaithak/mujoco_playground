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
"""GRU-based neural network dynamics model.

Architecture:
  Input  : history of (state, action) pairs, shape (T, state_dim + action_dim)
  GRU    : processes temporal sequence
  Dense  : maps GRU output to delta-state prediction
"""

from typing import Tuple

import jax
from flax import nnx


class GRUDynamicsNetwork(nnx.Module):
  """GRU-based neural network for dynamics prediction.

  The network takes a fixed-length history of (state, action) pairs and
  predicts the *delta* to the state at the next timestep.

  States with angle dimensions should be pre-transformed to (cos, sin)
  before being passed in, so the output delta is also in (cos_δ, sin_δ)
  format for those slots.
  """

  def __init__(
      self,
      hidden_size: int,
      state_dim: int,
      action_dim: int,
      *,
      rngs: nnx.Rngs,
  ):
    """Initialise the GRU dynamics network.

    Args:
      hidden_size: Size of GRU hidden state.
      state_dim: Dimension of (transformed) state vector.
      action_dim: Dimension of action / control vector.
      rngs: Flax NNX random number generators.
    """
    self.hidden_size = hidden_size
    self.state_dim = state_dim
    self.action_dim = action_dim

    input_dim = state_dim + action_dim
    self.gru_cell = nnx.GRUCell(input_dim, hidden_size, rngs=rngs)
    self.dense1 = nnx.Linear(hidden_size, hidden_size // 2, rngs=rngs)
    self.dense2 = nnx.Linear(hidden_size // 2, state_dim, rngs=rngs)

  def __call__(
      self,
      state_action_history: jax.Array,
      gru_state: jax.Array,
      deterministic: bool = False,
  ) -> Tuple[jax.Array, jax.Array]:
    """Forward pass.

    Args:
      state_action_history: Shape (history_length, state_dim + action_dim).
      gru_state: Hidden state, shape (hidden_size,).
      deterministic: Unused; kept for API compatibility.

    Returns:
      predicted_delta: Shape (state_dim,).
      new_gru_state: Updated GRU hidden state, shape (hidden_size,).
    """
    carry = gru_state
    for t in range(state_action_history.shape[0]):
      carry, _ = self.gru_cell(carry, state_action_history[t])

    x = nnx.relu(self.dense1(carry))
    delta = self.dense2(x)
    return delta, carry
