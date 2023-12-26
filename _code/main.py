
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
import gymnax

from models import NeuralODE, NeuralNetwork
from ode import train_dynamics
from helpers import rollout_parallel

def rollout_on_real_system(key, params, model_static, env_params, n_trajectories):
    # Rollout 
    keys_rollout = jr.split(key, n_trajectories)
    obss, actionss, rewardss, _, _= rollout_parallel(keys_rollout, params, model_static, env_params)
    # Return data
    data = jnp.concatenate([obss, jnp.expand_dims(actionss, axis=-1)], axis=2)
    # Get length of rollout
    ts = jnp.linspace(0, env_params.dt * env_params.max_steps_in_episode, env_params.max_steps_in_episode)

    return ts, data, rewardss

def main():
    SEED = 0
    key_policy, key_ODE, key_loop = jr.split(jr.PRNGKey(SEED), 3)

    # Policy Training Parameters
    policy_init_params = {
        'input_size' : 3,
        'output_size' : 3,
        'width_size' : 65,
        'depth' : 1,
        'key' : key_policy
    }

    # ODE Training Parameters
    dynamics_init_params = {
        'input_size' : 4,
        'output_size' : 3,
        'width_size' : 21,
        'depth' : 3,
        'key' : key_policy
    }

    len_strat = (0.15, )

    # Init Policy 
    policy = NeuralNetwork(**policy_init_params)
    # Init ODE
    dynamics = NeuralODE(**dynamics_init_params)

    # Loop training params
    real_env_dataset_size = 1000
    _, env_params = gymnax.make('Pendulum-v1')

    # Loop
    n_loops = 5
    key_loops = jr.split(key_loop, n_loops)
    for key_loop in key_loops:
        # Run agent with policy on real environment
        policy_params, policy_static = eqx.partition(policy, eqx.is_inexact_array)
        ts, real_env_data, rewardss = rollout_on_real_system(key_loop, policy_params, policy_static, env_params, n_trajectories=real_env_dataset_size)
        # Train dynamics model
        dynamics_params, dynamics_static = eqx.partition(dynamics, eqx.is_inexact_array)
        _, dynamics_losses, dynamics = train_dynamics(real_env_data, ts, dynamics_params, dynamics_static, key=key_ODE,
                                            length_strategy=len_strat)
        # Train policy on dynamics model 
        



if __name__ == '__main__':
    main()
