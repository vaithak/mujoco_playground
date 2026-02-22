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
"""CSV logger for MuJoCo/MJX rollout state-action trajectories.

Records one row per timestep:
  timestamp, qpos_0, ..., qvel_0, ..., ctrl_0, ...

Usage::

    logger = CSVLogger(output_dir="data_csv/cartpole", prefix="episode")
    logger.reset()
    for t in range(episode_length):
        logger.log(timestamp=t * dt, qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    logger.save()   # writes  data_csv/cartpole/episode_000.csv
"""

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np


class CSVLogger:
  """Accumulates state-action rows and saves them as a CSV file.

  Args:
    output_dir: Directory where CSV files will be written.
    prefix: Filename prefix; files are named ``{prefix}_{episode:03d}.csv``.
  """

  def __init__(self, output_dir: str, prefix: str = "episode"):
    self._dir = Path(output_dir)
    self._prefix = prefix
    self._dir.mkdir(parents=True, exist_ok=True)
    self._rows: list = []
    self._episode: int = 0
    self._header: Optional[list] = None

  # ------------------------------------------------------------------

  def reset(self) -> None:
    """Clear accumulated rows (call at the start of each episode)."""
    self._rows = []
    self._header = None

  def log(
      self,
      timestamp: float,
      qpos: np.ndarray,
      qvel: np.ndarray,
      ctrl: np.ndarray,
  ) -> None:
    """Append one timestep row.

    Args:
      timestamp: Simulation time in seconds.
      qpos: Position vector (any numpy-compatible array).
      qvel: Velocity vector.
      ctrl: Control / action vector.
    """
    qpos = np.asarray(qpos).flatten()
    qvel = np.asarray(qvel).flatten()
    ctrl = np.asarray(ctrl).flatten()

    if self._header is None:
      self._header = (
          ["timestamp"]
          + [f"qpos_{i}" for i in range(len(qpos))]
          + [f"qvel_{i}" for i in range(len(qvel))]
          + [f"ctrl_{i}" for i in range(len(ctrl))]
      )

    row = [timestamp] + qpos.tolist() + qvel.tolist() + ctrl.tolist()
    self._rows.append(row)

  def save(self) -> Optional[Path]:
    """Write accumulated rows to a CSV file and increment episode counter.

    Returns:
      Path to the written file, or None if there were no rows to save.
    """
    if not self._rows:
      return None  # nothing to save

    fname = self._dir / f"{self._prefix}_{self._episode:04d}.csv"
    with open(fname, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(self._header)
      writer.writerows(self._rows)

    print(f"Saved {len(self._rows)} rows → {fname}")
    self._episode += 1
    self.reset()
    return fname

  @property
  def num_rows(self) -> int:
    """Number of rows accumulated since last reset."""
    return len(self._rows)
