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
"""Utilities for training dynamics models."""

from typing import List, Tuple

import jax
import jax.numpy as jp
import optax
from flax.training import train_state

from mujoco_playground._src.learned_dynamics.simple_dynamics_model import SimpleDynamicsModel


def collect_rollout_data(
    env,
    policy,
    num_episodes: int = 100,
    rng_key: jax.Array,
    max_episode_length: int = 1000,
):
  """Collect state-action-next_state transitions from environment rollouts.

  Args:
      env: Environment to collect data from
      policy: Policy to use for action selection (can be random)
      num_episodes: Number of episodes to collect
      rng_key: Random key for initialization
      max_episode_length: Maximum length of each episode

  Returns:
      List of (qpos, qvel, action, next_qpos, next_qvel) tuples
  """
  transitions = []

  for _ in range(num_episodes):
    rng_key, reset_key, episode_key = jax.random.split(rng_key, 3)
    state = env.reset(reset_key)

    # Run episode
    for step in range(max_episode_length):
      episode_key, action_key = jax.random.split(episode_key)

      # Get action from policy
      action = policy(state, action_key)

      # Take step
      next_state = env.step(state, action)

      # Store transition
      transitions.append(
          (
              state.data.qpos,
              state.data.qvel,
              action,
              next_state.data.qpos,
              next_state.data.qvel,
          )
      )

      state = next_state

      # Check if done
      if next_state.done > 0.5:
        break

  return transitions


def train_dynamics_model(
    train_data: List[Tuple],
    hidden_dims: Tuple[int, ...] = (256, 256),
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 256,
    rng_key: jax.Array,
):
  """Train a dynamics model on collected transition data.

  Args:
      train_data: List of (qpos, qvel, action, next_qpos, next_qvel) transitions
      hidden_dims: Hidden layer dimensions for MLP
      learning_rate: Learning rate for optimizer
      num_epochs: Number of training epochs
      batch_size: Batch size for training
      rng_key: Random key for initialization

  Returns:
      Tuple of (trained model parameters, hidden_dims)
  """
  # Initialize model
  model = SimpleDynamicsModel(hidden_dims=hidden_dims)

  # Create dummy input for initialization
  qpos_example, qvel_example, action_example, _, _ = train_data[0]
  rng_key, init_key = jax.random.split(rng_key)
  params = model.init(init_key, qpos_example, qvel_example, action_example)

  # Create optimizer
  optimizer = optax.adam(learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=optimizer
  )

  # Define loss function (MSE between predicted and actual next state)
  def loss_fn(params, qpos, qvel, action, target_qpos, target_qvel):
    pred_qpos, pred_qvel = model.apply(params, qpos, qvel, action)
    loss_qpos = jp.mean((pred_qpos - target_qpos) ** 2)
    loss_qvel = jp.mean((pred_qvel - target_qvel) ** 2)
    return loss_qpos + loss_qvel

  # Training step
  @jax.jit
  def train_step(state, batch):
    qpos, qvel, action, target_qpos, target_qvel = batch

    def batch_loss(params):
      # Vectorized loss over batch
      losses = jax.vmap(lambda q, qv, a, tq, tqv: loss_fn(params, q, qv, a, tq, tqv))(
          qpos, qvel, action, target_qpos, target_qvel
      )
      return jp.mean(losses)

    loss, grads = jax.value_and_grad(batch_loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

  # Prepare data
  qpos_data = jp.array([t[0] for t in train_data])
  qvel_data = jp.array([t[1] for t in train_data])
  action_data = jp.array([t[2] for t in train_data])
  target_qpos_data = jp.array([t[3] for t in train_data])
  target_qvel_data = jp.array([t[4] for t in train_data])

  num_samples = len(train_data)

  # Training loop
  for epoch in range(num_epochs):
    # Shuffle data
    rng_key, shuffle_key = jax.random.split(rng_key)
    perm = jax.random.permutation(shuffle_key, num_samples)
    qpos_shuffled = qpos_data[perm]
    qvel_shuffled = qvel_data[perm]
    action_shuffled = action_data[perm]
    target_qpos_shuffled = target_qpos_data[perm]
    target_qvel_shuffled = target_qvel_data[perm]

    # Mini-batch training
    epoch_loss = 0.0
    num_batches = 0
    for i in range(0, num_samples, batch_size):
      batch = (
          qpos_shuffled[i : i + batch_size],
          qvel_shuffled[i : i + batch_size],
          action_shuffled[i : i + batch_size],
          target_qpos_shuffled[i : i + batch_size],
          target_qvel_shuffled[i : i + batch_size],
      )
      state, loss = train_step(state, batch)
      epoch_loss += loss
      num_batches += 1

    if (epoch + 1) % 10 == 0:
      avg_loss = epoch_loss / num_batches
      print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

  return state.params, hidden_dims
