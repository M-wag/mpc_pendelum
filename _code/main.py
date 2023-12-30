# %%
import jax
import jax.random as jr
import jax.numpy as jnp

import dill
import os
import equinox as eqx
import gymnax
import optax
from functools import partial
from flax import struct
from gymnax.environments.classic_control import Pendulum
from datetime import datetime

from models import NeuralODE, NeuralNetwork
from ode import train_dynamics
from helpers import rollout_parallel
from reinforce import train_step, train_step_ode

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_disable_jit", True)

print(f"Current Back End : {jax.lib.xla_bridge.get_backend().platform}", end='\n\n')

root_file_directory =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@eqx.filter_jit
def rollout_on_real_system(key, model, env_params, n_trajectories):
    params, model_static = eqx.partition(model, eqx.is_inexact_array)
    # Rollout 
    keys_rollout = jr.split(key, n_trajectories)
    obss, actionss, rewardss, _, _= rollout_parallel(keys_rollout, params, model_static, env_params)
    # Return data
    data = jnp.concatenate([obss, jnp.expand_dims(actionss, axis=-1)], axis=2)
    # Get length of rollout
    ts = jnp.linspace(0, env_params.dt * env_params.max_steps_in_episode, env_params.max_steps_in_episode)

    return ts, data, rewardss

def train_policy(policy_params, policy_static, env_params, key):
    key_train = key
    # Training parameters
    learning_rate = 5e-3
    n_runs = 3000
    n_batches = 64
    # Setup optimizer
    optim = optax.adam(learning_rate)
    opt_state = optim.init(policy_params)

    keys_train = jr.split(key_train, n_runs)
    (trained_params, _), losses = jax.lax.scan(
        partial(
            train_step,
            model_static=policy_static,
            env_params=env_params,
            optimizer=optim,
            n_batches=n_batches,
        ),
        (policy_params, opt_state),
        keys_train
    )

    return losses, eqx.combine(trained_params, policy_static)

def train_policy_on_ode(policy_params, policy_static, dynamics, env_params, key,
                        n_runs=2000):
    key_train = key
    # Training parameters
    learning_rate = 5e-3
    n_batches = 64
    # Setup optimizer
    optim = optax.adam(learning_rate)
    opt_state = optim.init(policy_params)

    keys_train = jr.split(key_train, n_runs)
    (trained_params, _, _), losses = jax.lax.scan(
        partial(
            train_step_ode,
            model_static=policy_static,
            dynamics=dynamics,
            env_params=env_params,
            optimizer=optim,
            n_batches=n_batches,
        ),
        (policy_params, opt_state, 0),
        keys_train
    )

    return losses, eqx.combine(trained_params, policy_static)



def main():
    SEED = 0
    key_policy, key_dynamics, key_loop = jr.split(jr.PRNGKey(SEED), 3)

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
        'key' : key_dynamics
    }

    # Init Policy 
    policy = NeuralNetwork(**policy_init_params)
    # Init ODE
    dynamics = NeuralODE(**dynamics_init_params)

    # Loop training params
    real_env_dataset_size = 2000
    _, env_params = gymnax.make('Pendulum-v1')
    key_loops = jr.split(key_loop, 5)

    dynamics_train_data = jnp.empty((0, 200, 4))
    time_start = datetime.now()
    for run, key_loop in enumerate(key_loops):
        # Split keys
        key_get_real_data, key_train_dynamics, key_train_policy = jr.split(key_loop, 3)

        # Run policy on real system
        print('Generating Dynamics Data...')
        ts, real_env_data, rewardss = rollout_on_real_system(key_get_real_data, policy, env_params, 
                                                                n_trajectories=real_env_dataset_size)

        dynamics_train_data = jnp.concatenate([dynamics_train_data, real_env_data], axis=0)
        # Train dynamics model
        print('Training Dynamics...')
        print(f'Dynamics Dataset Shape : {dynamics_train_data.shape}')
        old_dynamics_params, old_dynamics_static = eqx.partition(dynamics, eqx.is_inexact_array)
        _, dynamics_losses, dynamics = train_dynamics(dynamics_train_data, ts, old_dynamics_params, old_dynamics_static, key=key_train_dynamics, 
                                                    lr_strategy=(3e-3, 3e-3, 3e-3), 
                                                    steps_strategy=(400, 400, 400),
                                                    length_strategy=(0.10, 0.50, 1.0))
        # Train policy on REAL model 
        print('Training Policy...')
        old_policy_params, old_policy_static = eqx.partition(policy, eqx.is_inexact_array)
        policy_losses, policy = train_policy_on_ode(old_policy_params, old_policy_static, dynamics, env_params, key_train_policy,
                                                        n_runs=1000)
        print

        if True:
            check_point = {
                'run' : run,
                'dynamics_losses': dynamics_losses,
                'dynamics': dynamics,
                'policy_losses': policy_losses,
                'policy': policy,
            }

            checkpoint_path = os.path.join(root_file_directory, 'checkpoints', f'{time_start.strftime("%H:%M:%S")}', f'{run}.pkl')
            if not os.path.exists(os.path.dirname(checkpoint_path)):

                os.mkdir(os.path.dirname(checkpoint_path))

            with open(checkpoint_path, 'wb') as file:
                dill.dump(check_point, file)           
    


if __name__ == '__main__':
    main()
