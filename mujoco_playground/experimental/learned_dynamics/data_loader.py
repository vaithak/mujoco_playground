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
"""Data loader for training dynamics models from logged CSV trajectories.

Expected CSV format (one row per timestep):
  timestamp, qpos_0, qpos_1, ..., qvel_0, qvel_1, ..., ctrl_0, ctrl_1, ...

The dataset handles:
  - Angle transformation: angle index → (cos, sin) pair
  - Velocity normalisation to [-1, 1] range
  - Sliding-window examples of length history_length + prediction_horizon
"""

from pathlib import Path
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


def load_csv_data(
    csv_path: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Load a single CSV file containing logged state-action data.

  Args:
    csv_path: Path to the CSV file.

  Returns:
    Tuple (qpos, qvel, ctrl):
      qpos: shape (T, nq)
      qvel: shape (T, nv)
      ctrl: shape (T, nu)
  """
  import csv as _csv

  with open(csv_path, newline="") as f:
    reader = _csv.DictReader(f)
    rows = list(reader)

  if not rows:
    raise ValueError(f"Empty CSV file: {csv_path}")

  header = list(rows[0].keys())
  qpos_cols = sorted([c for c in header if c.startswith("qpos_")])
  qvel_cols = sorted([c for c in header if c.startswith("qvel_")])
  ctrl_cols = sorted([c for c in header if c.startswith("ctrl_")])

  qpos = np.array([[float(r[c]) for c in qpos_cols] for r in rows])
  qvel = np.array([[float(r[c]) for c in qvel_cols] for r in rows])
  ctrl = np.array([[float(r[c]) for c in ctrl_cols] for r in rows])

  return jnp.array(qpos), jnp.array(qvel), jnp.array(ctrl)


class DynamicsDataset:
  """Windowed dataset for training dynamics models.

  Each example is a tensor of shape
  ``(history_length + prediction_horizon, transformed_state_control_dim)``
  containing consecutive timesteps with:
    - transformed qpos (angles → cos/sin)
    - normalised qvel
    - ctrl (raw)

  Coordinate normalisation (relative to first history step) is *not* applied
  here; it is applied inside the training loss.

  Args:
    data_dir: Directory containing ``*.csv`` trajectory files.
    history_length: Number of past timesteps used as conditioning window.
    prediction_horizon: Number of future timesteps to predict.
    coordinate_indices: Indices in qpos that are Cartesian coordinates
      (used only to report ``transformed_coord_indices``; normalisation is
      done at train time).
    angle_indices: Indices in qpos that are angles (will be expanded to
      cos/sin pairs).
    normalize_velocities: If True normalise qvel to [-1, 1].
  """

  def __init__(
      self,
      data_dir: str,
      history_length: int = 11,
      prediction_horizon: int = 9,
      coordinate_indices: Optional[jnp.ndarray] = None,
      angle_indices: Optional[jnp.ndarray] = None,
      normalize_velocities: bool = True,
  ):
    self.data_dir = Path(data_dir)
    self.history_length = history_length
    self.prediction_horizon = prediction_horizon
    self.coordinate_indices = (
        coordinate_indices if coordinate_indices is not None else jnp.array([])
    )
    self.angle_indices = (
        angle_indices if angle_indices is not None else jnp.array([])
    )
    self.normalize_velocities = normalize_velocities

    if len(self.angle_indices) > 0:
      self.sorted_angle_indices: List[int] = [
          int(i) for i in jnp.sort(self.angle_indices)
      ]
    else:
      self.sorted_angle_indices = []

    self.trajectories = self._load_all_trajectories()

    if self.normalize_velocities:
      self.qvel_min, self.qvel_max = self._compute_velocity_stats()
      self.qvel_range = jnp.maximum(self.qvel_max - self.qvel_min, 1e-6)
    else:
      self.qvel_min = self.qvel_max = self.qvel_range = None

    sample_qpos = self.trajectories[0][0][0]
    transformed = self._transform_qpos(sample_qpos)
    self.transformed_qpos_dim = int(transformed.shape[-1])
    self.nv = int(self.trajectories[0][1].shape[1])
    self.nu = int(self.trajectories[0][2].shape[1])
    self.transformed_state_control_dim = (
        self.transformed_qpos_dim + self.nv + self.nu
    )
    self.transformed_coord_indices = self._compute_transformed_coord_indices()

    self.examples = self._create_examples()

    print(f"Loaded {len(self.trajectories)} trajectories from {self.data_dir}")
    print(f"Created {len(self.examples)} training windows")
    print(
        f"State+ctrl dim: {self.transformed_state_control_dim} "
        f"(qpos_t={self.transformed_qpos_dim}, qvel={self.nv}, ctrl={self.nu})"
    )

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _load_all_trajectories(
      self,
  ) -> List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    csv_files = sorted(self.data_dir.glob("*.csv"))
    if not csv_files:
      raise ValueError(f"No CSV files found in {self.data_dir}")
    return [load_csv_data(str(f)) for f in csv_files]

  def _compute_velocity_stats(
      self,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    all_qvel = jnp.concatenate([t[1] for t in self.trajectories], axis=0)
    return jnp.min(all_qvel, axis=0), jnp.max(all_qvel, axis=0)

  def _normalize_qvel(self, qvel: jnp.ndarray) -> jnp.ndarray:
    if not self.normalize_velocities:
      return qvel
    return (qvel - self.qvel_min) / self.qvel_range * 2.0 - 1.0

  def denormalize_qvel(self, qvel_norm: jnp.ndarray) -> jnp.ndarray:
    """Inverse of _normalize_qvel."""
    if not self.normalize_velocities:
      return qvel_norm
    return (qvel_norm + 1.0) / 2.0 * self.qvel_range + self.qvel_min

  def _compute_transformed_coord_indices(self) -> List[int]:
    if len(self.coordinate_indices) == 0:
      return []
    result = []
    for ci in self.coordinate_indices:
      extra = sum(
          1 for ai in self.sorted_angle_indices if ai < int(ci)
      )
      result.append(int(ci) + extra)
    return result

  def _transform_qpos(self, qpos: jnp.ndarray) -> jnp.ndarray:
    """Convert angles to (cos, sin); leave other entries unchanged."""
    if not self.sorted_angle_indices:
      return qpos
    parts = []
    prev = 0
    for ai in self.sorted_angle_indices:
      if ai > prev:
        parts.append(qpos[..., prev:ai])
      angle = qpos[..., ai]
      parts.append(jnp.expand_dims(jnp.cos(angle), -1))
      parts.append(jnp.expand_dims(jnp.sin(angle), -1))
      prev = ai + 1
    if prev < qpos.shape[-1]:
      parts.append(qpos[..., prev:])
    return jnp.concatenate(parts, axis=-1)

  def _create_examples(self) -> List[jnp.ndarray]:
    window = self.history_length + self.prediction_horizon
    examples = []
    for qpos_t, qvel_t, ctrl_t in self.trajectories:
      T = qpos_t.shape[0]
      for t in range(T - window + 1):
        qp = self._transform_qpos(qpos_t[t : t + window])
        qv = self._normalize_qvel(qvel_t[t : t + window])
        ct = ctrl_t[t : t + window]
        examples.append(jnp.concatenate([qp, qv, ct], axis=-1))
    return examples

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, idx: int) -> jnp.ndarray:
    return self.examples[idx]

  def get_batch(self, indices: jnp.ndarray) -> jnp.ndarray:
    """Return stacked examples for the given indices."""
    return jnp.stack([self.examples[int(i)] for i in indices])

  def split(
      self, train_ratio: float = 0.8, seed: int = 42
  ) -> Tuple["DynamicsDataset", "DynamicsDataset"]:
    """Split into train / validation datasets."""
    n_train = int(len(self.examples) * train_ratio)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(self.examples))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    def _clone(indices):
      ds = DynamicsDataset.__new__(DynamicsDataset)
      ds.__dict__.update(self.__dict__)
      ds.examples = [self.examples[i] for i in indices]
      return ds

    return _clone(train_idx), _clone(val_idx)
