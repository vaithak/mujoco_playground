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
"""Evaluation utilities for trained GRU dynamics models.

Provides:
  - load_trained_model(checkpoint_path) → (network, config)
  - evaluate_model(network, dataset, config) → metrics dict
  - plot_training_history(history)
  - plot_prediction_errors(metrics)
  - evaluate_from_checkpoint(checkpoint_path, data_dir, ...)  (CLI entry)
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from mujoco_playground.experimental.learned_dynamics.data_loader import (
    DynamicsDataset,
)
from mujoco_playground.experimental.learned_dynamics.network import (
    GRUDynamicsNetwork,
)
from mujoco_playground.experimental.learned_dynamics.train import (
    apply_delta_to_state,
    normalize_history,
)


def load_trained_model(
    checkpoint_path: str,
) -> Tuple[GRUDynamicsNetwork, Dict[str, Any]]:
  """Load a trained GRU dynamics model from an Orbax checkpoint directory.

  Args:
    checkpoint_path: Directory produced by train_dynamics_model, containing
      ``state/`` (Orbax) and ``config.pkl``.

  Returns:
    (network, config) where config is the dict saved during training.
  """
  import orbax.checkpoint as ocp
  from flax import nnx

  p = Path(checkpoint_path)

  with open(p / "config.pkl", "rb") as f:
    payload = pickle.load(f)
  config = payload["config"]

  if config["coordinate_indices"] is not None:
    config["coordinate_indices"] = jnp.array(config["coordinate_indices"])
  if config["angle_indices"] is not None:
    config["angle_indices"] = jnp.array(config["angle_indices"])

  network = GRUDynamicsNetwork(
      hidden_size=config["hidden_size"],
      state_dim=config["state_dim"],
      action_dim=config["action_dim"],
      rngs=nnx.Rngs(0),
  )
  checkpointer = ocp.PyTreeCheckpointer()
  _, state = nnx.split(network)
  restored = checkpointer.restore(p / "state", item=state)
  graph_def, _ = nnx.split(network)
  network = nnx.merge(graph_def, restored)

  return network, config


def evaluate_model(
    network: GRUDynamicsNetwork,
    dataset: DynamicsDataset,
    config: Dict[str, Any],
    num_examples: int = 100,
) -> Dict[str, Any]:
  """Compute per-step MSE and MAE over *num_examples* validation windows.

  Args:
    network: Trained GRU dynamics network.
    dataset: Dataset (typically validation split).
    config: Config dict from load_trained_model.
    num_examples: How many windows to evaluate.

  Returns:
    Dict with keys ``mse_per_step``, ``mae_per_step``,
    ``avg_mse_per_step``, ``avg_mae_per_step``.
  """
  hidden_size = config["hidden_size"]
  tqpos_dim = config["transformed_qpos_dim"]
  history_length = config["history_length"]
  prediction_horizon = config["prediction_horizon"]
  sorted_angles = dataset.sorted_angle_indices
  coord_idx = dataset.transformed_coord_indices
  state_dim = tqpos_dim + dataset.nv
  coord_arr = jnp.array(coord_idx)
  gru0 = jnp.zeros((hidden_size,))

  mse_rows, mae_rows = [], []
  num_examples = min(num_examples, len(dataset))

  for i in range(num_examples):
    seq = dataset[i]
    state_history = seq[:history_length]
    step_mse, step_mae = [], []

    for h in range(prediction_horizon):
      norm_hist = normalize_history(state_history, coord_arr, state_dim)
      delta, _ = network(norm_hist, gru0, deterministic=True)
      last_state = state_history[-1, :state_dim]
      next_state = apply_delta_to_state(
          last_state, delta, tqpos_dim, sorted_angles
      )
      gt = seq[history_length + h, :state_dim]
      step_mse.append(float(jnp.mean(jnp.square(next_state - gt))))
      step_mae.append(float(jnp.mean(jnp.abs(next_state - gt))))

      if h < prediction_horizon - 1:
        next_ctrl = seq[history_length + h, state_dim:]
        state_history = jnp.concatenate(
            [
                state_history[1:],
                jnp.expand_dims(jnp.concatenate([next_state, next_ctrl]), 0),
            ],
            axis=0,
        )

    mse_rows.append(step_mse)
    mae_rows.append(step_mae)

  avg_mse = jnp.mean(jnp.array(mse_rows), axis=0)
  avg_mae = jnp.mean(jnp.array(mae_rows), axis=0)

  return {
      "mse_per_step": mse_rows,
      "mae_per_step": mae_rows,
      "avg_mse_per_step": avg_mse,
      "avg_mae_per_step": avg_mae,
  }


def plot_training_history(
    history: Dict[str, Any], save_path: Optional[str] = None
) -> None:
  """Plot train / val loss curves.

  Args:
    history: Dict with ``train_loss`` and ``val_loss`` lists.
    save_path: If given, save figure to this path instead of showing.
  """
  import matplotlib.pyplot as plt

  plt.figure(figsize=(9, 5))
  plt.plot(history["train_loss"], label="Train", linewidth=2)
  plt.plot(history["val_loss"], label="Val", linewidth=2)
  plt.xlabel("Epoch")
  plt.ylabel("MSE Loss")
  plt.title("Training History")
  plt.legend()
  plt.grid(alpha=0.3)
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training history plot → {save_path}")
  else:
    plt.show()
  plt.close()


def plot_prediction_errors(
    metrics: Dict[str, Any], save_path: Optional[str] = None
) -> None:
  """Plot per-step MSE and MAE over the prediction horizon.

  Args:
    metrics: Dict from evaluate_model.
    save_path: If given, save figure instead of showing.
  """
  import matplotlib.pyplot as plt

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
  steps = range(1, len(metrics["avg_mse_per_step"]) + 1)

  ax1.plot(steps, metrics["avg_mse_per_step"], marker="o", linewidth=2)
  ax1.set_xlabel("Prediction Step")
  ax1.set_ylabel("MSE")
  ax1.set_title("MSE over Horizon")
  ax1.grid(alpha=0.3)

  ax2.plot(
      steps, metrics["avg_mae_per_step"], marker="o", linewidth=2, color="C1"
  )
  ax2.set_xlabel("Prediction Step")
  ax2.set_ylabel("MAE")
  ax2.set_title("MAE over Horizon")
  ax2.grid(alpha=0.3)

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved prediction error plot → {save_path}")
  else:
    plt.show()
  plt.close()


def evaluate_from_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    num_examples: int = 100,
) -> Dict[str, Any]:
  """Load a checkpoint, build dataset, run evaluation and optionally save plots.

  Args:
    checkpoint_path: Path to checkpoint directory.
    data_dir: Directory containing CSV data.
    output_dir: If given, save plots and JSON metrics here.
    num_examples: Number of validation windows to evaluate.

  Returns:
    Metrics dict from evaluate_model.
  """
  print(f"Loading model from {checkpoint_path} …")
  network, config = load_trained_model(checkpoint_path)

  print(f"Loading dataset from {data_dir} …")
  dataset = DynamicsDataset(
      data_dir=data_dir,
      history_length=config["history_length"],
      prediction_horizon=config["prediction_horizon"],
      coordinate_indices=config.get("coordinate_indices"),
      angle_indices=config.get("angle_indices"),
  )
  _, val_ds = dataset.split(train_ratio=0.8)

  print(f"Evaluating on {num_examples} examples …")
  metrics = evaluate_model(network, val_ds, config, num_examples)

  print("\nAverage MSE per step:")
  for i, v in enumerate(metrics["avg_mse_per_step"]):
    print(f"  step {i+1}: {v:.6f}")
  print("\nAverage MAE per step:")
  for i, v in enumerate(metrics["avg_mae_per_step"]):
    print(f"  step {i+1}: {v:.6f}")

  if output_dir:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "metrics.json", "w") as f:
      json.dump(
          {
              "avg_mse_per_step": [
                  float(x) for x in metrics["avg_mse_per_step"]
              ],
              "avg_mae_per_step": [
                  float(x) for x in metrics["avg_mae_per_step"]
              ],
          },
          f,
          indent=2,
      )

    with open(Path(checkpoint_path) / "config.pkl", "rb") as cfg_f:
      payload = pickle.load(cfg_f)
    if "history" in payload:
      plot_training_history(
          payload["history"], save_path=str(out / "training_history.png")
      )
    plot_prediction_errors(metrics, save_path=str(out / "prediction_errors.png"))
    print(f"\nResults saved to {output_dir}")

  return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Evaluate a trained dynamics model")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--data-dir", required=True)
  parser.add_argument("--output-dir", default=None)
  parser.add_argument("--num-examples", type=int, default=100)
  args = parser.parse_args()

  evaluate_from_checkpoint(
      checkpoint_path=args.checkpoint,
      data_dir=args.data_dir,
      output_dir=args.output_dir,
      num_examples=args.num_examples,
  )
