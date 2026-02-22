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
"""Training script for GRU dynamics models.

Implements an autoregressive multi-step prediction loss:
  1. Initialise with history_length actual timesteps.
  2. For each prediction step h = 0 … prediction_horizon-1:
       a. Normalise coordinate indices of current history relative to t=0.
       b. Feed normalised history through GRUDynamicsNetwork → delta.
       c. Apply delta to un-normalised last state using trig formulas for
          angle slots (cos/sin representation).
       d. Append predicted state + next ground-truth control to history.
  3. MSE between predicted states and ground-truth states.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from mujoco_playground.experimental.learned_dynamics.data_loader import (
    DynamicsDataset,
)
from mujoco_playground.experimental.learned_dynamics.network import (
    GRUDynamicsNetwork,
)


# ---------------------------------------------------------------------------
# State-space helpers
# ---------------------------------------------------------------------------


def apply_delta_to_state(
    current_state: jnp.ndarray,
    delta: jnp.ndarray,
    transformed_qpos_dim: int,
    sorted_angle_indices: List[int],
) -> jnp.ndarray:
  """Apply predicted delta to current (transformed) state.

  For angle slots (cos/sin representation), uses trigonometric addition:
    cos(θ+δ) = cos θ · cos δ − sin θ · sin δ
    sin(θ+δ) = sin θ · cos δ + cos θ · sin δ

  Args:
    current_state: Shape (state_dim,) — transformed qpos ++ qvel.
    delta: Shape (state_dim,).
    transformed_qpos_dim: Dimension of the transformed qpos section.
    sorted_angle_indices: Original (pre-transform) indices of angle DOFs.

  Returns:
    Next state, shape (state_dim,).
  """
  cur_qpos = current_state[:transformed_qpos_dim]
  cur_qvel = current_state[transformed_qpos_dim:]
  d_qpos = delta[:transformed_qpos_dim]
  d_qvel = delta[transformed_qpos_dim:]

  next_qvel = cur_qvel + d_qvel

  if not sorted_angle_indices:
    next_qpos = cur_qpos + d_qpos
  else:
    parts = []
    t_idx = 0  # index in transformed space
    o_idx = 0  # index in original qpos
    for ai in sorted_angle_indices:
      seg = ai - o_idx
      if seg > 0:
        parts.append(cur_qpos[t_idx : t_idx + seg] + d_qpos[t_idx : t_idx + seg])
        t_idx += seg
      # trig update for (cos, sin) pair
      cos_t, sin_t = cur_qpos[t_idx], cur_qpos[t_idx + 1]
      cos_d, sin_d = d_qpos[t_idx], d_qpos[t_idx + 1]
      parts.append(
          jnp.array([
              cos_t * cos_d - sin_t * sin_d,
              sin_t * cos_d + cos_t * sin_d,
          ])
      )
      t_idx += 2
      o_idx = ai + 1
    if t_idx < transformed_qpos_dim:
      parts.append(cur_qpos[t_idx:] + d_qpos[t_idx:])
    next_qpos = jnp.concatenate(parts)

  return jnp.concatenate([next_qpos, next_qvel])


def normalize_history(
    history: jnp.ndarray,
    coordinate_indices: jnp.ndarray,
    state_dim: int,
) -> jnp.ndarray:
  """Subtract first-step coordinate values from all timesteps in history.

  Args:
    history: Shape (history_length, state_dim + action_dim).
    coordinate_indices: Indices in the *transformed* state to normalise.
    state_dim: Dimension of the state portion.

  Returns:
    Normalised history with the same shape.
  """
  if len(coordinate_indices) == 0:
    return history
  origin = history[0, coordinate_indices]
  normalized = history
  for i, ci in enumerate(coordinate_indices):
    normalized = normalized.at[:, ci].add(-origin[i])
  return normalized


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_autoregressive_loss(
    network: GRUDynamicsNetwork,
    sequence: jnp.ndarray,
    history_length: int,
    prediction_horizon: int,
    transformed_qpos_dim: int,
    sorted_angle_indices: List[int],
    transformed_coord_indices: List[int],
    hidden_size: int,
    state_dim: int,
) -> jnp.ndarray:
  """MSE loss over autoregressive multi-step predictions.

  Args:
    network: GRU dynamics network.
    sequence: Shape (history_length + prediction_horizon, state_dim + action_dim).
    history_length: Conditioning window length.
    prediction_horizon: Number of future steps.
    transformed_qpos_dim: Dim of transformed qpos.
    sorted_angle_indices: Pre-sorted angle indices (original qpos space).
    transformed_coord_indices: Coordinate indices after angle expansion.
    hidden_size: GRU hidden size (for zero initialisation).
    state_dim: Dimension of state (= transformed_qpos_dim + nv).

  Returns:
    Scalar MSE loss.
  """
  state_history = sequence[:history_length]
  gru0 = jnp.zeros((hidden_size,))
  coord_arr = jnp.array(transformed_coord_indices)

  predicted_states = []
  for h in range(prediction_horizon):
    norm_hist = normalize_history(state_history, coord_arr, state_dim)
    delta, _ = network(norm_hist, gru0, deterministic=False)
    last_state = state_history[-1, :state_dim]
    next_state = apply_delta_to_state(
        last_state, delta, transformed_qpos_dim, sorted_angle_indices
    )
    predicted_states.append(next_state)

    if h < prediction_horizon - 1:
      next_ctrl = sequence[history_length + h, state_dim:]
      next_sa = jnp.concatenate([next_state, next_ctrl])
      state_history = jnp.concatenate(
          [state_history[1:], jnp.expand_dims(next_sa, 0)], axis=0
      )

  pred = jnp.stack(predicted_states, axis=0)
  gt = sequence[history_length : history_length + prediction_horizon, :state_dim]
  return jnp.mean(jnp.square(pred - gt))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    network: GRUDynamicsNetwork,
    checkpoint_path: Path,
    config: Dict[str, Any],
    history: Dict[str, Any],
) -> None:
  """Save network state + config to *checkpoint_path* (Orbax + pickle)."""
  import orbax.checkpoint as ocp

  checkpoint_path.mkdir(parents=True, exist_ok=True)
  graph_def, state = nnx.split(network)
  checkpointer = ocp.PyTreeCheckpointer()
  checkpointer.save(checkpoint_path / "state", state, force=True)

  payload = {"config": config, "history": history}
  with open(checkpoint_path / "config.pkl", "wb") as f:
    pickle.dump(payload, f)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def train_dynamics_model(
    data_dir: str,
    task_name: str,
    coordinate_indices: Optional[jnp.ndarray] = None,
    angle_indices: Optional[jnp.ndarray] = None,
    hidden_size: int = 128,
    history_length: int = 11,
    prediction_horizon: int = 9,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_dir: Optional[str] = None,
    seed: int = 0,
) -> Tuple[GRUDynamicsNetwork, Dict[str, Any]]:
  """Train a GRU dynamics model on logged CSV trajectory data.

  Args:
    data_dir: Directory containing ``*.csv`` files.
    task_name: Used for checkpoint sub-directory naming.
    coordinate_indices: Cartesian coordinate indices in qpos.
    angle_indices: Angle indices in qpos (will be cos/sin expanded).
    hidden_size: GRU hidden dimension.
    history_length: Length of conditioning window.
    prediction_horizon: Autoregressive prediction steps.
    batch_size: Mini-batch size.
    num_epochs: Training epochs.
    learning_rate: AdamW learning rate.
    weight_decay: AdamW weight decay.
    checkpoint_dir: Root checkpoint directory (default: ``data_dir/../checkpoints``).
    seed: Random seed.

  Returns:
    (trained_network, training_history) where training_history has keys
    ``train_loss`` and ``val_loss`` (lists of floats per epoch).
  """
  key = jax.random.PRNGKey(seed)

  print(f"Loading data from {data_dir} …")
  dataset = DynamicsDataset(
      data_dir=data_dir,
      history_length=history_length,
      prediction_horizon=prediction_horizon,
      coordinate_indices=coordinate_indices,
      angle_indices=angle_indices,
  )
  train_ds, val_ds = dataset.split(train_ratio=0.8, seed=seed)
  print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

  state_dim = dataset.transformed_qpos_dim + dataset.nv
  action_dim = dataset.nu

  network = GRUDynamicsNetwork(
      hidden_size=hidden_size,
      state_dim=state_dim,
      action_dim=action_dim,
      rngs=nnx.Rngs(seed),
  )
  optimizer = nnx.Optimizer(
      model=network,
      tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
      wrt=nnx.Param,
  )

  sorted_angle_indices = dataset.sorted_angle_indices
  coord_indices = dataset.transformed_coord_indices
  tqpos_dim = dataset.transformed_qpos_dim

  # JIT-compiled step functions
  @nnx.jit
  def train_step(model, batch):
    def loss_fn(m):
      def single(seq):
        return compute_autoregressive_loss(
            m, seq, history_length, prediction_horizon,
            tqpos_dim, sorted_angle_indices, coord_indices,
            hidden_size, state_dim,
        )
      return jnp.mean(nnx.vmap(single)(batch))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads

  @nnx.jit
  def val_step(model, batch):
    def single(seq):
      return compute_autoregressive_loss(
          model, seq, history_length, prediction_horizon,
          tqpos_dim, sorted_angle_indices, coord_indices,
          hidden_size, state_dim,
      )
    return jnp.mean(nnx.vmap(single)(batch))

  if checkpoint_dir is None:
    checkpoint_dir = str(Path(data_dir).parent / "checkpoints")
  ckpt_root = Path(checkpoint_dir) / task_name
  ckpt_root.mkdir(parents=True, exist_ok=True)

  cfg = {
      "hidden_size": hidden_size,
      "state_dim": state_dim,
      "action_dim": action_dim,
      "history_length": history_length,
      "prediction_horizon": prediction_horizon,
      "coordinate_indices": (
          coordinate_indices.tolist() if coordinate_indices is not None else None
      ),
      "angle_indices": (
          angle_indices.tolist() if angle_indices is not None else None
      ),
      "transformed_qpos_dim": tqpos_dim,
      "normalize_velocities": dataset.normalize_velocities,
      "qvel_min": (
          dataset.qvel_min.tolist() if dataset.qvel_min is not None else None
      ),
      "qvel_max": (
          dataset.qvel_max.tolist() if dataset.qvel_max is not None else None
      ),
      "qvel_range": (
          dataset.qvel_range.tolist()
          if dataset.qvel_range is not None
          else None
      ),
  }

  history: Dict[str, Any] = {"train_loss": [], "val_loss": []}
  best_val = float("inf")

  print(f"\nTraining for {num_epochs} epochs …  checkpoints → {ckpt_root}")

  for epoch in range(num_epochs):
    # --- training ---
    key, sk = jax.random.split(key)
    perm = jax.random.permutation(sk, len(train_ds))
    n_batches = len(train_ds) // batch_size
    train_losses = []

    with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
      for i in range(n_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        batch = train_ds.get_batch(idx)
        loss, grads = train_step(network, batch)
        optimizer.update(network, grads)
        train_losses.append(float(loss))
        pbar.set_postfix({"loss": f"{loss:.5f}"})
        pbar.update(1)

    avg_train = float(jnp.mean(jnp.array(train_losses)))
    history["train_loss"].append(avg_train)

    # --- validation ---
    val_losses = []
    n_val = len(val_ds) // batch_size
    for i in range(n_val):
      idx = jnp.arange(i * batch_size, (i + 1) * batch_size)
      batch = val_ds.get_batch(idx)
      val_losses.append(float(val_step(network, batch)))
    avg_val = float(jnp.mean(jnp.array(val_losses))) if val_losses else float("inf")
    history["val_loss"].append(avg_val)

    print(
        f"Epoch {epoch+1}/{num_epochs}  "
        f"train={avg_train:.5f}  val={avg_val:.5f}"
    )

    if avg_val < best_val:
      best_val = avg_val
      _save_checkpoint(network, ckpt_root / "best_model", cfg, history)
      print(f"  → saved best model (val={best_val:.5f})")

    if (epoch + 1) % 50 == 0:
      _save_checkpoint(
          network, ckpt_root / f"model_epoch_{epoch+1}", cfg, history
      )

  print(f"\nTraining complete. Best val loss: {best_val:.5f}")
  print(f"Best model saved to: {ckpt_root / 'best_model'}")
  return network, history
