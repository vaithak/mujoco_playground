# Learned Dynamics Training

A self-contained package for training GRU-based neural network dynamics models
on logged state-action trajectory data from `mujoco_playground` environments.

Inspired by and compatible with the approach in
[vaithak/hydrax](https://github.com/vaithak/hydrax) at commit `8e2afa2`.

---

## Directory Structure

```
learned_dynamics/
├── __init__.py         Public API
├── network.py          GRU dynamics network (Flax NNX)
├── data_loader.py      CSV loader + DynamicsDataset
├── train.py            train_dynamics_model() entrypoint
├── evaluate.py         Checkpoint loader + evaluation + plots
├── csv_logger.py       CSVLogger for rollout data collection
├── README.md           This file
└── examples/
    ├── collect_data.py     Generate CSV datasets (random / PPO)
    ├── train_cartpole.py   Train dynamics model – CartpoleSwingup
    ├── train_acrobot.py    Train dynamics model – AcrobotSwingup
    └── interactive_play.py Live MuJoCo viewer with optional CSV logging
```

---

## Quick Start

### 1. Collect trajectory data

```bash
# 50 random-policy episodes for CartpoleSwingup
python examples/collect_data.py \
    --task CartpoleSwingup \
    --policy random \
    --num-episodes 50 \
    --episode-length 500

# With a trained PPO policy
python examples/collect_data.py \
    --task CartpoleSwingup \
    --policy ppo \
    --checkpoint logs/CartpoleSwingup-.../checkpoints \
    --num-episodes 50

# Acrobot
python examples/collect_data.py \
    --task AcrobotSwingup \
    --policy random \
    --num-episodes 50
```

Data is written to `data_csv/<task_lower>/<policy>/episode_NNNN.csv`.

### 2. Train a dynamics model

```bash
# Cartpole
python examples/train_cartpole.py \
    --data-dir data_csv/cartpoleswingup/random \
    --num-epochs 200

# Acrobot
python examples/train_acrobot.py \
    --data-dir data_csv/acrobotswingup/random \
    --num-epochs 200
```

Checkpoints are saved to `data_csv/<task>/checkpoints/<task_name>/`.

### 3. Launch interactive viewer

```bash
# Random policy, CartpoleSwingup, live GLFW window
python examples/interactive_play.py --task CartpoleSwingup

# PPO policy with CSV logging
python examples/interactive_play.py \
    --task CartpoleSwingup \
    --policy ppo \
    --checkpoint logs/CartpoleSwingup-.../checkpoints \
    --log-csv \
    --output-dir data_csv/cartpoleswingup/interactive
```

---

## CSV Format

Each CSV file contains one row per timestep:

```
timestamp,qpos_0,...,qvel_0,...,ctrl_0,...
0.00,0.0,0.0,0.0,0.0,1.0
0.01,0.002,0.003,0.02,0.03,0.9
...
```

| Column group | Description |
|---|---|
| `timestamp` | Simulation time in seconds |
| `qpos_i` | Position DOF `i` (raw, not transformed) |
| `qvel_i` | Velocity DOF `i` |
| `ctrl_i` | Control / action DOF `i` |

---

## Coordinate and Angle Indices

The `coordinate_indices` and `angle_indices` parameters control how the
dataset preprocesses state vectors:

- **`angle_indices`**: positions in `qpos` that are angles. Each is expanded
  to a `(cos, sin)` pair, increasing the transformed state dimension by one
  per angle.
- **`coordinate_indices`**: Cartesian coordinate positions in `qpos`
  (e.g. a cart's x-position). These are normalised *at training time* relative
  to the first step in each history window.

### CartpoleSwingup

```
qpos = [cart_x (0), pole_angle (1)]
qvel = [cart_x_vel (0), pole_angle_vel (1)]
ctrl = [cart_force (0)]

coordinate_indices = jnp.array([0])   # cart_x
angle_indices      = jnp.array([1])   # pole_angle
```

### AcrobotSwingup

```
qpos = [upper_arm_angle (0), lower_arm_angle (1)]
qvel = [upper_arm_vel (0), lower_arm_vel (1)]
ctrl = [joint_torque (0)]

coordinate_indices = None             # no Cartesian coords
angle_indices      = jnp.array([0, 1])  # both are angles
```

---

## Programmatic API

```python
import jax.numpy as jnp
from mujoco_playground.experimental.learned_dynamics import (
    train_dynamics_model,
    evaluate_from_checkpoint,
    CSVLogger,
)

# Train
network, history = train_dynamics_model(
    data_dir="data_csv/cartpoleswingup/random",
    task_name="cartpole",
    coordinate_indices=jnp.array([0]),
    angle_indices=jnp.array([1]),
    hidden_size=64,
    history_length=4,
    prediction_horizon=15,
    num_epochs=200,
)

# Evaluate
metrics = evaluate_from_checkpoint(
    checkpoint_path="data_csv/cartpole/checkpoints/cartpole/best_model",
    data_dir="data_csv/cartpoleswingup/random",
    output_dir="results/cartpole_eval",
)

# CSV logging
logger = CSVLogger(output_dir="data_csv/cartpoleswingup/manual", prefix="ep")
logger.reset()
logger.log(timestamp=0.0, qpos=[0.0, 0.0], qvel=[0.0, 0.0], ctrl=[1.0])
logger.save()
```

---

## Model Architecture

```
Input: (history_length, state_dim + action_dim)
  └─ GRUCell × history_length  → hidden (hidden_size,)
       └─ Linear(hidden_size, hidden_size//2) + ReLU
            └─ Linear(hidden_size//2, state_dim)
                 → delta (state_dim,)
```

Angles are stored as `(cos, sin)` pairs; deltas for angle slots are also
`(cos_δ, sin_δ)` and applied via trigonometric addition formulas.

---

## Training Loss

Autoregressive multi-step prediction:

1. Initialise `state_history` with first `history_length` ground-truth steps.
2. For each step `h` in `[0, prediction_horizon)`:
   a. Normalise coordinate indices relative to `state_history[0]`.
   b. Run GRU on normalised history → `delta`.
   c. Apply `delta` to the *un-normalised* last state → `next_state`.
   d. Append `(next_state, ground_truth_ctrl)` to history.
3. Loss = mean MSE across all prediction steps.

---

## Dependencies

Uses the existing `mujoco_playground` stack:
- `jax`, `flax` (NNX), `optax`, `orbax-checkpoint`, `tqdm`
- `pandas`, `numpy` (data loading)
- `matplotlib` (optional, for plots)
- `mujoco` with GLFW support (interactive viewer)
