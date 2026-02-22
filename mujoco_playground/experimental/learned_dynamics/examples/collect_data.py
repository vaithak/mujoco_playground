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
"""Collect CSV trajectory data for cartpole or acrobot.

Supports two collection modes:
  --policy random    – uniform random actions
  --policy ppo       – run a trained PPO checkpoint (requires --checkpoint)

Output is written to:
  data_csv/<task_name>/random/  or  data_csv/<task_name>/ppo/

Usage examples::

  # Collect 50 random-policy episodes for CartpoleSwingup
  python collect_data.py --task CartpoleSwingup --policy random \
      --num-episodes 50 --episode-length 500

  # Collect 50 PPO episodes for CartpoleSwingup
  python collect_data.py --task CartpoleSwingup --policy ppo \
      --checkpoint logs/CartpoleSwingup-.../checkpoints \
      --num-episodes 50 --episode-length 500

  # Acrobot random
  python collect_data.py --task AcrobotSwingup --policy random \
      --num-episodes 50 --episode-length 500
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import jax
import numpy as np

# Suppress verbose JAX/MuJoCo logs
os.environ.setdefault("MUJOCO_GL", "egl")


def _load_ppo_policy(checkpoint_path: str, env, seed: int = 0):
  """Load a PPO policy from a Brax checkpoint directory.

  Tries to load the latest numeric sub-directory (Orbax format used by
  ``learning/train_jax_ppo.py``).

  Returns:
    jit-compiled inference_fn(obs, rng) -> (action, extras)
  """
  from brax.training.agents.ppo import networks as ppo_networks
  from etils import epath

  ckpt = epath.Path(checkpoint_path).resolve()
  if ckpt.is_dir():
    subdirs = sorted(
        [d for d in ckpt.iterdir() if d.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else 0,
    )
    if subdirs:
      ckpt = subdirs[-1]

  obs_size = env.observation_size
  action_size = env.action_size

  network_factory = ppo_networks.make_ppo_networks
  ppo_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      preprocess_observations_fn=lambda x, _: x,
  )
  make_policy = ppo_networks.make_inference_fn(ppo_network)

  import orbax.checkpoint as ocp

  checkpointer = ocp.PyTreeCheckpointer()
  params = checkpointer.restore(ckpt)
  inference_fn = make_policy(params, deterministic=True)
  return jax.jit(inference_fn)


def collect_episodes(
    env_name: str,
    policy: str,
    num_episodes: int,
    episode_length: int,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    seed: int = 0,
) -> None:
  """Collect *num_episodes* rollouts and save each as a CSV file.

  Args:
    env_name: Registry name, e.g. ``"CartpoleSwingup"``.
    policy: ``"random"`` or ``"ppo"``.
    num_episodes: Number of episodes to collect.
    episode_length: Steps per episode.
    output_dir: Directory for CSV output.
    checkpoint_path: Path to PPO checkpoint (required when policy="ppo").
    seed: Base RNG seed.
  """
  from mujoco_playground import registry
  from mujoco_playground.experimental.learned_dynamics import CSVLogger

  env = registry.load(env_name)
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  inference_fn = None
  if policy == "ppo":
    if checkpoint_path is None:
      raise ValueError("--checkpoint is required when --policy ppo")
    print(f"Loading PPO policy from {checkpoint_path} …")
    inference_fn = _load_ppo_policy(checkpoint_path, env, seed)

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  logger = CSVLogger(output_dir=str(out), prefix="episode")
  rng = jax.random.PRNGKey(seed)

  print(
      f"Collecting {num_episodes} episodes × {episode_length} steps "
      f"({policy} policy) → {out}"
  )

  for ep in range(num_episodes):
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    logger.reset()

    for t in range(episode_length):
      rng, act_rng = jax.random.split(rng)

      if policy == "random":
        action = jax.random.uniform(
            act_rng,
            shape=(env.action_size,),
            minval=-1.0,
            maxval=1.0,
        )
      else:
        action = inference_fn(state.obs, act_rng)[0]

      logger.log(
          timestamp=float(state.data.time),
          qpos=np.asarray(state.data.qpos),
          qvel=np.asarray(state.data.qvel),
          ctrl=np.asarray(action),
      )
      state = jit_step(state, action)

    logger.save()
    if (ep + 1) % 10 == 0:
      print(f"  {ep+1}/{num_episodes} episodes done")

  print(f"Done. {num_episodes} CSV files written to {out}")


def main():
  parser = argparse.ArgumentParser(
      description="Collect CSV trajectory data from mujoco_playground envs"
  )
  parser.add_argument(
      "--task",
      default="CartpoleSwingup",
      help='Registry env name (default: CartpoleSwingup)',
  )
  parser.add_argument(
      "--policy",
      choices=["random", "ppo"],
      default="random",
      help="Policy type (default: random)",
  )
  parser.add_argument(
      "--checkpoint",
      default=None,
      help="Path to PPO checkpoint directory (required when --policy ppo)",
  )
  parser.add_argument(
      "--num-episodes",
      type=int,
      default=50,
      help="Number of episodes to collect (default: 50)",
  )
  parser.add_argument(
      "--episode-length",
      type=int,
      default=500,
      help="Steps per episode (default: 500)",
  )
  parser.add_argument(
      "--output-dir",
      default=None,
      help=(
          "Output directory for CSV files. "
          "Defaults to data_csv/<task>/<policy>/"
      ),
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  if args.output_dir is None:
    args.output_dir = str(
        Path("data_csv") / args.task.lower() / args.policy
    )

  collect_episodes(
      env_name=args.task,
      policy=args.policy,
      num_episodes=args.num_episodes,
      episode_length=args.episode_length,
      output_dir=args.output_dir,
      checkpoint_path=args.checkpoint,
      seed=args.seed,
  )


if __name__ == "__main__":
  main()
