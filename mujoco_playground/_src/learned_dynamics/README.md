# Learned Dynamics for Model-Based RL

This module enables model-based reinforcement learning by replacing MJX physics simulation with learned neural network dynamics models.

## Usage

See `examples/cartpole_learned_dynamics_demo.py` for a complete example.

### Basic Usage

```python
import jax
from mujoco_playground import registry
from mujoco_playground._src.learned_dynamics import (
    LearnedDynamicsEnv,
    SimpleDynamicsModel,
    train_dynamics_model,
    collect_rollout_data,
    create_dynamics_model_fn,
)

# 1. Load environment
env = registry.load('CartpoleBalance')

# 2. Collect rollout data (using random policy)
def random_policy(state, rng):
    return jax.random.uniform(rng, (env.action_size,), minval=-1, maxval=1)

rng_key = jax.random.PRNGKey(0)
rollout_data = collect_rollout_data(env, random_policy, rng_key, num_episodes=100)

# 3. Train dynamics model
rng_key, train_key = jax.random.split(rng_key)
trained_params, hidden_dims = train_dynamics_model(rollout_data, train_key)

# 4. Create learned dynamics environment
dynamics_fn = create_dynamics_model_fn(trained_params, hidden_dims)
learned_env = LearnedDynamicsEnv(env, dynamics_fn)

# 5. Use like a normal environment
state = learned_env.reset(jax.random.PRNGKey(0))
action = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,))
next_state = learned_env.step(state, action)
```

## Components

### `LearnedDynamicsEnv`
Environment wrapper that uses learned dynamics instead of MJX physics simulation.
- Maintains full compatibility with the base environment interface
- JIT-compilable for performance
- Supports all standard environment operations (reset, step, render)

### `SimpleDynamicsModel`
MLP-based neural network for learning state transitions.
- Predicts state deltas (next_state - current_state) for training stability
- Configurable hidden layer dimensions
- Uses ReLU activations

### `train_dynamics_model`
Training utilities for dynamics models.
- Supports mini-batch training
- Uses Adam optimizer
- MSE loss for state prediction
- Returns both trained parameters and hidden dimensions

### `collect_rollout_data`
Utilities for collecting training data from environment rollouts.
- Collects (state, action, next_state) transitions
- Supports any policy (random, trained, etc.)

## Benefits

- **Sim-to-Real Transfer**: Train dynamics on real robot data for better real-world performance
- **Faster Rollouts**: Potentially faster than full physics simulation if model is smaller
- **Differentiable Dynamics**: Enable gradient-based planning and model-based RL algorithms
- **Model-Based RL**: Use learned models for planning, imagination, or world model training

## Implementation Notes

- The dynamics model predicts state deltas rather than absolute states for training stability
- All components are JAX-compatible and JIT-compilable
- The wrapper maintains full compatibility with Brax PPO and other training frameworks
- Observations and rewards are computed using the original environment's methods
