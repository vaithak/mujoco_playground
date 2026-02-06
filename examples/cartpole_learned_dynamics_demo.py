#!/usr/bin/env python3
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
"""Demo of using learned dynamics for CartPole environment.

This example demonstrates:
1. Loading the CartPole environment
2. Collecting rollout data from the real MJX environment
3. Training a simple dynamics model on the collected data
4. Creating a LearnedDynamicsEnv wrapper with the trained model
5. Running rollouts with the learned dynamics
6. Comparing performance between real and learned dynamics
"""

import jax
import jax.numpy as jp

from mujoco_playground._src import registry
from mujoco_playground._src.learned_dynamics import (
    LearnedDynamicsEnv,
    collect_rollout_data,
    create_dynamics_model_fn,
    train_dynamics_model,
)


def random_policy(state, rng):
  """Simple random policy for data collection.

  Args:
      state: Current environment state
      rng: Random number generator key

  Returns:
      Random action in [-1, 1]
  """
  return jax.random.uniform(rng, (1,), minval=-1.0, maxval=1.0)


def run_episode(env, policy, rng_key, max_steps=1000):
  """Run a single episode and collect states.

  Args:
      env: Environment to run
      policy: Policy function
      rng_key: Random key
      max_steps: Maximum episode length

  Returns:
      List of states from the episode
  """
  rng_key, reset_key = jax.random.split(rng_key)
  state = env.reset(reset_key)
  states = [state]

  for _ in range(max_steps):
    rng_key, action_key = jax.random.split(rng_key)
    action = policy(state, action_key)
    state = env.step(state, action)
    states.append(state)

    if state.done > 0.5:
      break

  return states


def compute_trajectory_difference(states_real, states_learned):
  """Compute difference between two trajectories.

  Args:
      states_real: List of states from real environment
      states_learned: List of states from learned environment

  Returns:
      Average position and velocity difference
  """
  min_len = min(len(states_real), len(states_learned))
  qpos_diffs = []
  qvel_diffs = []

  for i in range(min_len):
    qpos_diff = jp.mean(jp.abs(states_real[i].data.qpos - states_learned[i].data.qpos))
    qvel_diff = jp.mean(jp.abs(states_real[i].data.qvel - states_learned[i].data.qvel))
    qpos_diffs.append(qpos_diff)
    qvel_diffs.append(qvel_diff)

  return jp.mean(jp.array(qpos_diffs)), jp.mean(jp.array(qvel_diffs))


def main():
  """Main demo function."""
  print("=" * 80)
  print("CartPole Learned Dynamics Demo")
  print("=" * 80)
  print()

  # 1. Load CartPole environment
  print("1. Loading CartPole environment...")
  env = registry.load('CartpoleBalance')
  print(f"   Environment loaded: {env.__class__.__name__}")
  print(f"   Action size: {env.action_size}")
  print(f"   Observation size: {env.observation_size}")
  print()

  # 2. Collect data from real environment
  print("2. Collecting rollout data from real environment...")
  print("   Using random policy for data collection...")
  rng_key = jax.random.PRNGKey(0)
  rollout_data = collect_rollout_data(
      env, random_policy, num_episodes=50, rng_key=rng_key
  )
  print(f"   Collected {len(rollout_data)} transitions")
  print()

  # 3. Train dynamics model
  print("3. Training dynamics model...")
  print("   Model architecture: MLP with hidden dims (256, 256)")
  print("   Training for 50 epochs with batch size 256...")
  rng_key, train_key = jax.random.split(rng_key)
  trained_params, hidden_dims = train_dynamics_model(
      rollout_data,
      hidden_dims=(256, 256),
      learning_rate=1e-3,
      num_epochs=50,
      batch_size=256,
      rng_key=train_key,
  )
  print("   Training complete!")
  print()

  # 4. Create learned dynamics environment
  print("4. Creating learned dynamics environment...")
  dynamics_fn = create_dynamics_model_fn(trained_params, hidden_dims)
  learned_env = LearnedDynamicsEnv(env, dynamics_fn)
  print("   Learned environment created!")
  print()

  # 5. Compare rollouts
  print("5. Running comparison rollouts...")
  print("   Running 5 test episodes with the same random seeds...")
  print()

  total_qpos_diff = 0.0
  total_qvel_diff = 0.0
  num_test_episodes = 5

  for episode_idx in range(num_test_episodes):
    rng_key, episode_key = jax.random.split(rng_key)

    # Run with real dynamics
    states_real = run_episode(env, random_policy, episode_key, max_steps=100)

    # Run with learned dynamics (same seed)
    states_learned = run_episode(learned_env, random_policy, episode_key, max_steps=100)

    # Compute difference
    qpos_diff, qvel_diff = compute_trajectory_difference(states_real, states_learned)
    total_qpos_diff += qpos_diff
    total_qvel_diff += qvel_diff

    print(f"   Episode {episode_idx + 1}:")
    print(f"     Real trajectory length: {len(states_real)}")
    print(f"     Learned trajectory length: {len(states_learned)}")
    print(f"     Avg position difference: {qpos_diff:.6f}")
    print(f"     Avg velocity difference: {qvel_diff:.6f}")

  print()
  print("   Average over all test episodes:")
  print(f"     Avg position difference: {total_qpos_diff / num_test_episodes:.6f}")
  print(f"     Avg velocity difference: {total_qvel_diff / num_test_episodes:.6f}")
  print()

  # 6. Summary
  print("=" * 80)
  print("Demo Summary:")
  print("=" * 80)
  print("✓ Successfully loaded CartPole environment")
  print(f"✓ Collected {len(rollout_data)} training transitions")
  print("✓ Trained neural network dynamics model")
  print("✓ Created LearnedDynamicsEnv wrapper")
  print("✓ Compared real vs learned dynamics trajectories")
  print()
  print("The learned dynamics model can now be used for:")
  print("  - Model-based reinforcement learning")
  print("  - Fast rollout generation for planning")
  print("  - Sim-to-real transfer (when trained on real robot data)")
  print("  - Gradient-based trajectory optimization")
  print()
  print("Done!")
  print("=" * 80)


if __name__ == '__main__':
  main()
