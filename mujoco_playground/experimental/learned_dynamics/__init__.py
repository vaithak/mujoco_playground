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
"""Learned dynamics training package for mujoco_playground.

Provides:
- GRU-based dynamics network (network.py)
- CSV data loader and DynamicsDataset (data_loader.py)
- Training entrypoint train_dynamics_model (train.py)
- Evaluation utilities (evaluate.py)
- CSV rollout logger (csv_logger.py)
"""

from mujoco_playground.experimental.learned_dynamics.csv_logger import CSVLogger
from mujoco_playground.experimental.learned_dynamics.data_loader import (
    DynamicsDataset,
    load_csv_data,
)
from mujoco_playground.experimental.learned_dynamics.evaluate import (
    evaluate_from_checkpoint,
    load_trained_model,
    plot_prediction_errors,
    plot_training_history,
)
from mujoco_playground.experimental.learned_dynamics.network import GRUDynamicsNetwork
from mujoco_playground.experimental.learned_dynamics.train import train_dynamics_model

__all__ = [
    "CSVLogger",
    "DynamicsDataset",
    "GRUDynamicsNetwork",
    "evaluate_from_checkpoint",
    "load_csv_data",
    "load_trained_model",
    "plot_prediction_errors",
    "plot_training_history",
    "train_dynamics_model",
]
