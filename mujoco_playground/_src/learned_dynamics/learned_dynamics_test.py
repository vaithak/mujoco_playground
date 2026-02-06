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
"""Tests for learned dynamics module."""

from absl.testing import absltest
import jax
import jax.numpy as jp

from mujoco_playground._src import dm_control_suite
from mujoco_playground._src.learned_dynamics import (
    LearnedDynamicsEnv,
    SimpleDynamicsModel,
    create_dynamics_model_fn,
    collect_rollout_data,
    train_dynamics_model,
)


class LearnedDynamicsTest(absltest.TestCase):
  """Test suite for learned dynamics module."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.env = dm_control_suite.load('CartpoleBalance')
    self.rng_key = jax.random.PRNGKey(0)

  def test_simple_dynamics_model_forward(self):
    """Test basic forward pass of SimpleDynamicsModel."""
    model = SimpleDynamicsModel(hidden_dims=(64, 64))
    rng_key = jax.random.PRNGKey(0)

    # Create dummy inputs
    qpos = jp.array([0.0, 0.1])
    qvel = jp.array([0.0, 0.0])
    action = jp.array([1.0])

    # Initialize and run model
    params = model.init(rng_key, qpos, qvel, action)
    next_qpos, next_qvel = model.apply(params, qpos, qvel, action)

    # Check outputs have correct shape
    self.assertEqual(next_qpos.shape, qpos.shape)
    self.assertEqual(next_qvel.shape, qvel.shape)

  def test_create_dynamics_model_fn(self):
    """Test creation of dynamics model function."""
    model = SimpleDynamicsModel(hidden_dims=(64, 64))
    rng_key = jax.random.PRNGKey(0)

    # Initialize model
    qpos = jp.array([0.0, 0.1])
    qvel = jp.array([0.0, 0.0])
    action = jp.array([1.0])
    params = model.init(rng_key, qpos, qvel, action)

    # Create dynamics function
    dynamics_fn = create_dynamics_model_fn(params, hidden_dims=(64, 64))

    # Test with mock state data
    state = self.env.reset(rng_key)
    next_data = dynamics_fn(state.data, action)

    # Check that data is updated
    self.assertEqual(next_data.qpos.shape, state.data.qpos.shape)
    self.assertEqual(next_data.qvel.shape, state.data.qvel.shape)

  def test_learned_env_reset(self):
    """Test that LearnedDynamicsEnv reset works correctly."""
    # Create a simple dynamics model
    model = SimpleDynamicsModel(hidden_dims=(32, 32))
    rng_key = jax.random.PRNGKey(0)

    # Initialize with dummy data
    state = self.env.reset(rng_key)
    action = jp.zeros(self.env.action_size)
    params = model.init(rng_key, state.data.qpos, state.data.qvel, action)

    # Create learned env
    dynamics_fn = create_dynamics_model_fn(params, hidden_dims=(32, 32))
    learned_env = LearnedDynamicsEnv(self.env, dynamics_fn)

    # Test reset
    reset_state = learned_env.reset(rng_key)

    # Check state structure
    self.assertIsNotNone(reset_state.data)
    self.assertIsNotNone(reset_state.obs)
    self.assertIsNotNone(reset_state.reward)
    self.assertIsNotNone(reset_state.done)
    self.assertIsNotNone(reset_state.metrics)
    self.assertIsNotNone(reset_state.info)

  def test_learned_env_step(self):
    """Test that LearnedDynamicsEnv step works correctly."""
    # Create a simple dynamics model
    model = SimpleDynamicsModel(hidden_dims=(32, 32))
    rng_key = jax.random.PRNGKey(0)

    # Initialize with dummy data
    state = self.env.reset(rng_key)
    action = jp.zeros(self.env.action_size)
    params = model.init(rng_key, state.data.qpos, state.data.qvel, action)

    # Create learned env
    dynamics_fn = create_dynamics_model_fn(params, hidden_dims=(32, 32))
    learned_env = LearnedDynamicsEnv(self.env, dynamics_fn)

    # Reset and step
    state = learned_env.reset(rng_key)
    next_state = learned_env.step(state, action)

    # Check state structure
    self.assertIsNotNone(next_state.data)
    self.assertIsNotNone(next_state.obs)
    self.assertIsNotNone(next_state.reward)
    self.assertIsNotNone(next_state.done)
    self.assertEqual(next_state.data.qpos.shape, state.data.qpos.shape)
    self.assertEqual(next_state.data.qvel.shape, state.data.qvel.shape)

  def test_learned_env_jit_compatible(self):
    """Test that LearnedDynamicsEnv is JIT-compilable."""
    # Create a simple dynamics model
    model = SimpleDynamicsModel(hidden_dims=(32, 32))
    rng_key = jax.random.PRNGKey(0)

    # Initialize with dummy data
    state = self.env.reset(rng_key)
    action = jp.zeros(self.env.action_size)
    params = model.init(rng_key, state.data.qpos, state.data.qvel, action)

    # Create learned env
    dynamics_fn = create_dynamics_model_fn(params, hidden_dims=(32, 32))
    learned_env = LearnedDynamicsEnv(self.env, dynamics_fn)

    # JIT compile step function
    jitted_step = jax.jit(learned_env.step)

    # Test that it works
    state = learned_env.reset(rng_key)
    next_state = jitted_step(state, action)

    self.assertIsNotNone(next_state)
    self.assertEqual(next_state.data.qpos.shape, state.data.qpos.shape)

  def test_learned_env_properties(self):
    """Test that LearnedDynamicsEnv exposes necessary properties."""
    # Create a simple dynamics model
    model = SimpleDynamicsModel(hidden_dims=(32, 32))
    rng_key = jax.random.PRNGKey(0)

    # Initialize with dummy data
    state = self.env.reset(rng_key)
    action = jp.zeros(self.env.action_size)
    params = model.init(rng_key, state.data.qpos, state.data.qvel, action)

    # Create learned env
    dynamics_fn = create_dynamics_model_fn(params, hidden_dims=(32, 32))
    learned_env = LearnedDynamicsEnv(self.env, dynamics_fn)

    # Check properties
    self.assertEqual(learned_env.action_size, self.env.action_size)
    self.assertEqual(learned_env.observation_size, self.env.observation_size)
    self.assertEqual(learned_env.dt, self.env.dt)
    self.assertIsNotNone(learned_env.xml_path)
    self.assertIsNotNone(learned_env.mj_model)
    self.assertIsNotNone(learned_env.mjx_model)

  def test_collect_rollout_data(self):
    """Test data collection from rollouts."""

    def random_policy(state, rng):
      return jax.random.uniform(rng, (self.env.action_size,), minval=-1, maxval=1)

    # Collect small amount of data
    data = collect_rollout_data(self.env, random_policy, self.rng_key, num_episodes=2)

    # Check data structure
    self.assertGreater(len(data), 0)
    qpos, qvel, action, next_qpos, next_qvel = data[0]
    self.assertEqual(qpos.shape, next_qpos.shape)
    self.assertEqual(qvel.shape, next_qvel.shape)
    self.assertEqual(action.shape, (self.env.action_size,))

  def test_train_dynamics_model(self):
    """Test training a dynamics model."""

    def random_policy(state, rng):
      return jax.random.uniform(rng, (self.env.action_size,), minval=-1, maxval=1)

    # Collect small amount of data
    data = collect_rollout_data(self.env, random_policy, self.rng_key, num_episodes=2)

    # Train model (just a few epochs for testing)
    params, hidden_dims = train_dynamics_model(
        data,
        self.rng_key,
        hidden_dims=(32, 32),
        learning_rate=1e-3,
        num_epochs=2,
        batch_size=32,
    )

    # Check that params exist
    self.assertIsNotNone(params)
    self.assertEqual(hidden_dims, (32, 32))

  def test_learned_env_completes_episode(self):
    """Test that learned env can complete an episode without errors."""
    # Create and train a simple model
    def random_policy(state, rng):
      return jax.random.uniform(rng, (self.env.action_size,), minval=-1, maxval=1)

    # Collect data
    data = collect_rollout_data(self.env, random_policy, self.rng_key, num_episodes=2)

    # Train model
    params, hidden_dims = train_dynamics_model(
        data, self.rng_key, hidden_dims=(32, 32), num_epochs=2, batch_size=32
    )

    # Create learned env
    dynamics_fn = create_dynamics_model_fn(params, hidden_dims)
    learned_env = LearnedDynamicsEnv(self.env, dynamics_fn)

    # Run a short episode
    rng_key = jax.random.PRNGKey(42)
    state = learned_env.reset(rng_key)

    for _ in range(10):
      rng_key, action_key = jax.random.split(rng_key)
      action = jax.random.uniform(action_key, (learned_env.action_size,), minval=-1, maxval=1)
      state = learned_env.step(state, action)

      # Episode should not have errors
      self.assertFalse(jp.isnan(state.data.qpos).any())
      self.assertFalse(jp.isnan(state.data.qvel).any())


if __name__ == '__main__':
  absltest.main()
