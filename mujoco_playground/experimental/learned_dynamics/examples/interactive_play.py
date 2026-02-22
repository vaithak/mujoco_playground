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
"""Interactive MuJoCo viewer for any mujoco_playground environment.

Runs a policy (random or loaded from a PPO checkpoint) indefinitely using
MuJoCo's native GLFW/passive viewer, rendering live. Optionally logs each
episode to CSV.

Requirements: ``mujoco`` must be installed with viewer support (the standard
pip package includes GLFW bindings).

Usage::

  # Random policy – CartpoleSwingup, live viewer, no CSV logging
  python interactive_play.py --task CartpoleSwingup

  # PPO policy, CartpoleSwingup, with CSV logging
  python interactive_play.py --task CartpoleSwingup \\
      --policy ppo \\
      --checkpoint logs/CartpoleSwingup-.../checkpoints \\
      --log-csv --output-dir data_csv/cartpole_interactive

  # Acrobot random
  python interactive_play.py --task AcrobotSwingup --episode-length 500
"""

import argparse
import os
import time
from typing import Optional

import jax
import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")


# ---------------------------------------------------------------------------
# PPO policy loader (mirrors collect_data.py)
# ---------------------------------------------------------------------------


def _load_ppo_policy(checkpoint_path: str, env, seed: int = 0):
  from brax.training.agents.ppo import networks as ppo_networks
  from etils import epath
  import orbax.checkpoint as ocp

  ckpt = epath.Path(checkpoint_path).resolve()
  if ckpt.is_dir():
    subdirs = sorted(
        [d for d in ckpt.iterdir() if d.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else 0,
    )
    if subdirs:
      ckpt = subdirs[-1]

  ppo_network = ppo_networks.make_ppo_networks(
      observation_size=env.observation_size,
      action_size=env.action_size,
      preprocess_observations_fn=lambda x, _: x,
  )
  make_policy = ppo_networks.make_inference_fn(ppo_network)
  checkpointer = ocp.PyTreeCheckpointer()
  params = checkpointer.restore(ckpt)
  inference_fn = make_policy(params, deterministic=True)
  return jax.jit(inference_fn)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def interactive_play(
    env_name: str,
    policy: str = "random",
    checkpoint_path: Optional[str] = None,
    episode_length: int = 1000,
    log_csv: bool = False,
    output_dir: Optional[str] = None,
    num_episodes: Optional[int] = None,
    seed: int = 0,
) -> None:
  """Run policy in an interactive MuJoCo viewer loop.

  Args:
    env_name: Registry environment name.
    policy: ``"random"`` or ``"ppo"``.
    checkpoint_path: Path to PPO checkpoint (required when policy="ppo").
    episode_length: Maximum steps per episode.
    log_csv: If True, write each episode to a CSV file.
    output_dir: Directory for CSV files (used only when log_csv=True).
    num_episodes: Stop after this many episodes (None = run forever).
    seed: RNG seed.
  """
  from mujoco_playground import registry
  from mujoco_playground.experimental.learned_dynamics import CSVLogger

  env = registry.load(env_name)
  mj_model = env.mj_model
  mj_data = mujoco.MjData(mj_model)

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  inference_fn = None
  if policy == "ppo":
    if checkpoint_path is None:
      raise ValueError("--checkpoint is required when --policy ppo")
    print(f"Loading PPO policy from {checkpoint_path} …")
    inference_fn = _load_ppo_policy(checkpoint_path, env, seed)

  logger = None
  if log_csv:
    if output_dir is None:
      output_dir = f"data_csv/{env_name.lower()}/interactive"
    logger = CSVLogger(output_dir=output_dir, prefix="episode")

  rng = jax.random.PRNGKey(seed)
  episode = 0

  print(f"\nStarting interactive viewer for {env_name} ({policy} policy)")
  print("Close the viewer window to stop.\n")

  with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
      if num_episodes is not None and episode >= num_episodes:
        break

      rng, rk = jax.random.split(rng)
      state = jit_reset(rk)
      if logger is not None:
        logger.reset()

      for t in range(episode_length):
        if not viewer.is_running():
          break

        rng, ak = jax.random.split(rng)
        if policy == "random":
          action = jax.random.uniform(
              ak, shape=(env.action_size,), minval=-1.0, maxval=1.0
          )
        else:
          action = inference_fn(state.obs, ak)[0]

        if logger is not None:
          logger.log(
              timestamp=float(state.data.time),
              qpos=np.asarray(state.data.qpos),
              qvel=np.asarray(state.data.qvel),
              ctrl=np.asarray(action),
          )

        state = jit_step(state, action)

        # Sync viewer with current state
        mj_data.qpos[:] = np.asarray(state.data.qpos)
        mj_data.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        # Respect real-time pacing approximately
        time.sleep(env.dt * 0.5)

      if logger is not None:
        logger.save()

      episode += 1
      print(f"Episode {episode} finished (t={t+1} steps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(
      description="Interactive MuJoCo viewer with optional CSV logging"
  )
  parser.add_argument(
      "--task",
      default="CartpoleSwingup",
      help="Registry env name (default: CartpoleSwingup)",
  )
  parser.add_argument(
      "--policy",
      choices=["random", "ppo"],
      default="random",
  )
  parser.add_argument(
      "--checkpoint",
      default=None,
      help="Path to PPO checkpoint (required for --policy ppo)",
  )
  parser.add_argument(
      "--episode-length",
      type=int,
      default=1000,
  )
  parser.add_argument(
      "--log-csv",
      action="store_true",
      help="Write each episode to a CSV file",
  )
  parser.add_argument(
      "--output-dir",
      default=None,
      help="Directory for CSV output (used with --log-csv)",
  )
  parser.add_argument(
      "--num-episodes",
      type=int,
      default=None,
      help="Stop after N episodes (default: run indefinitely)",
  )
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  interactive_play(
      env_name=args.task,
      policy=args.policy,
      checkpoint_path=args.checkpoint,
      episode_length=args.episode_length,
      log_csv=args.log_csv,
      output_dir=args.output_dir,
      num_episodes=args.num_episodes,
      seed=args.seed,
  )


if __name__ == "__main__":
  main()
