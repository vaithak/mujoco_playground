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
"""Environment wrapper that uses learned dynamics instead of MJX simulation."""

from typing import Any, Callable, Dict

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import mjx_env


class LearnedDynamicsEnv:
  """Environment wrapper that uses learned dynamics instead of MJX simulation.

  This wrapper replaces the physics simulation step with predictions from
  a learned neural network model, enabling model-based RL and sim-to-real
  transfer.
  """

  def __init__(
      self, base_env: mjx_env.MjxEnv, dynamics_model: Callable[[mjx.Data, jax.Array], mjx.Data]
  ):
    """Initialize the learned dynamics environment.

    Args:
        base_env: The base MJX environment to wrap
        dynamics_model: A callable that takes (state_data, action) and returns
          next state_data
    """
    self._base_env = base_env
    self._dynamics_model = dynamics_model

    # Copy necessary attributes from base env for compatibility
    self._ctrl_dt = base_env._ctrl_dt
    self._sim_dt = base_env._sim_dt

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset using the base environment's reset.

    Args:
        rng: Random number generator key

    Returns:
        Initial state from base environment
    """
    return self._base_env.reset(rng)

  def step(
      self, state: mjx_env.State, action: jax.Array
  ) -> mjx_env.State:
    """Step using learned dynamics instead of MJX physics.

    Args:
        state: Current environment state
        action: Action to take

    Returns:
        Next state with predicted dynamics
    """
    # Use learned model to predict next state
    next_data = self._dynamics_model(state.data, action)

    # Apply mjx.forward to update derived quantities (xpos, xmat, etc.)
    next_data = mjx.forward(self._base_env.mjx_model, next_data)

    # Compute reward using base env's reward function if available
    # We need to call the environment's internal reward method
    reward = jp.array(0.0)
    metrics = state.metrics
    info = state.info

    # Try to get reward from base env if it has the necessary methods
    if hasattr(self._base_env, '_get_reward'):
      reward = self._base_env._get_reward(next_data, action, info, metrics)
    elif hasattr(self._base_env, '_dense_reward'):
      reward = self._base_env._dense_reward(next_data, action, info, metrics)

    # Compute observation using base env's observation function
    obs = state.obs  # Default to previous obs
    if hasattr(self._base_env, '_get_obs'):
      obs = self._base_env._get_obs(next_data, info)

    # Check termination conditions (NaN detection)
    done = jp.isnan(next_data.qpos).any() | jp.isnan(next_data.qvel).any()
    done = done.astype(float)

    return mjx_env.State(next_data, obs, reward, done, metrics, info)

  def render(self, trajectory, **kwargs):
    """Delegate rendering to base environment.

    Args:
        trajectory: List of states or single state to render
        **kwargs: Additional arguments passed to base env render

    Returns:
        Rendered frames from base environment
    """
    return self._base_env.render(trajectory, **kwargs)

  @property
  def xml_path(self) -> str:
    """Path to the xml file for the environment."""
    return self._base_env.xml_path

  @property
  def action_size(self) -> int:
    """Size of the action space."""
    return self._base_env.action_size

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    """Size of the observation space."""
    return self._base_env.observation_size

  @property
  def mj_model(self):
    """Mujoco model for the environment."""
    return self._base_env.mj_model

  @property
  def mjx_model(self):
    """Mjx model for the environment."""
    return self._base_env.mjx_model

  @property
  def dt(self) -> float:
    """Control timestep for the environment."""
    return self._ctrl_dt

  @property
  def sim_dt(self) -> float:
    """Simulation timestep for the environment."""
    return self._sim_dt

  @property
  def n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.dt / self.sim_dt))

  @property
  def model_assets(self) -> Dict[str, Any]:
    """Dictionary of model assets."""
    return self._base_env.model_assets

  @property
  def unwrapped(self) -> mjx_env.MjxEnv:
    """Get the base unwrapped environment."""
    return self._base_env.unwrapped
