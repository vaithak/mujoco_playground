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
"""Train a GRU dynamics model for the AcrobotSwingup task.

Acrobot configuration
---------------------
  qpos: [upper_arm_angle (0), lower_arm_angle (1)]
  qvel: [upper_arm_vel (0),   lower_arm_vel (1)]
  ctrl: [joint_torque (0)]

  coordinate_indices = []      # no Cartesian coordinates
  angle_indices      = [0, 1]  # both joints are angles

Usage::

  # 1. Collect data first (see collect_data.py):
  python collect_data.py --task AcrobotSwingup --num-episodes 50

  # 2. Train:
  python train_acrobot.py \
      --data-dir data_csv/acrobotswingup/random \
      --num-epochs 200
"""

import argparse
from pathlib import Path

import jax.numpy as jnp


def main():
  parser = argparse.ArgumentParser(
      description="Train GRU dynamics model for AcrobotSwingup"
  )
  parser.add_argument(
      "--data-dir",
      default="data_csv/acrobotswingup/random",
      help="Directory containing CSV trajectory files",
  )
  parser.add_argument(
      "--checkpoint-dir",
      default=None,
      help="Root checkpoint directory (default: data_csv/../checkpoints)",
  )
  parser.add_argument("--hidden-size", type=int, default=64)
  parser.add_argument("--history-length", type=int, default=6)
  parser.add_argument("--prediction-horizon", type=int, default=15)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--num-epochs", type=int, default=200)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-5)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  from mujoco_playground.experimental.learned_dynamics import train_dynamics_model
  from mujoco_playground.experimental.learned_dynamics import plot_training_history

  # Both joint angles need cos/sin; no Cartesian coordinates
  coordinate_indices = None  # no Cartesian coords
  angle_indices = jnp.array([0, 1])

  cfg = dict(
      data_dir=args.data_dir,
      task_name="acrobot",
      coordinate_indices=coordinate_indices,
      angle_indices=angle_indices,
      hidden_size=args.hidden_size,
      history_length=args.history_length,
      prediction_horizon=args.prediction_horizon,
      batch_size=args.batch_size,
      num_epochs=args.num_epochs,
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
      seed=args.seed,
  )
  if args.checkpoint_dir is not None:
    cfg["checkpoint_dir"] = args.checkpoint_dir

  print("=" * 60)
  print("Training Dynamics Model – AcrobotSwingup")
  print("=" * 60)
  for k, v in cfg.items():
    print(f"  {k}: {v}")
  print()

  network, history = train_dynamics_model(**cfg)

  if args.checkpoint_dir is not None:
    ckpt_root = Path(args.checkpoint_dir) / "acrobot"
  else:
    ckpt_root = Path(args.data_dir).parent / "checkpoints" / "acrobot"

  plot_training_history(
      history,
      save_path=str(ckpt_root / "training_history.png"),
  )

  print("\n" + "=" * 60)
  print(f"Done. Best model → {ckpt_root / 'best_model'}")
  print("=" * 60)


if __name__ == "__main__":
  main()
