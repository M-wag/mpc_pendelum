# %%

import jax
import jax.random as jr
import equinox as eqx
import optax
import gymnax
from functools import partial
import time 

from reinforce import rollout_ode_parallel, rollout_parallel, rollout_ode
from models import NeuralNetwork, NeuralODE
from reinforce import train_step_ode
from helpers import rollout

def train_policy_on_ode(policy_params, policy_static, dynamics, env_params, key, n_runs=2000):
    key_train = key
    # Training parameters
    learning_rate = 5e-3
    n_batches = 32
    # Setup optimizer
    optim = optax.adam(learning_rate)
    opt_state = optim.init(policy_params)

    keys_train = jr.split(key_train, n_runs)
    (trained_params, _), losses = jax.lax.scan(
        partial(
            train_step_ode,
            model_static=policy_static,
            dynamics=dynamics,
            env_params=env_params,
            optimizer=optim,
            n_batches=n_batches,
        ),
        (policy_params, opt_state),
        keys_train
    )
    return losses, eqx.combine(trained_params, policy_static)

if __name__ == '__main__':
    SEED = 1
    key_policy, key_dynamics, key_train_pol, key_rollout = jr.split(jr.PRNGKey(SEED), 4)
    _, env_params = gymnax.make('Pendulum-v1')
    
    policy_params, policy_static = eqx.partition(NeuralNetwork(3, 3, 65, 1, key=key_policy), eqx.is_inexact_array)
    dynamics = NeuralODE(4, 3, 22, 3, key=key_dynamics)

    # for i in range(100_000):
    #     batch_size = 64 * 4
    #     keys_rollout  = jr.split(key_rollout, batch_size)

    #     # start_time = time.time()
    #     # rollout_parallel(keys_rollout, policy_params, policy_static, env_params)
    #     # print(f"Rollout Parallel Gymnax {batch_size} batch_size : {time.time() - start_time}")

    #     start_time = time.time()
    #     rollout_ode_parallel(keys_rollout, eqx.combine(policy_params, policy_static), dynamics, env_params)
    #     # rollout_ode(key_rollout, eqx.combine(policy_params, policy_static), dynamics, env_params)
    #     print(f" Run {i} : {time.time() - start_time}")

    for n_runs in [2, 10 ,50, 100, 200]:
        start_time = time.time()
        policy_losses, policy = train_policy_on_ode(policy_params, policy_static, dynamics, env_params, key_train_pol,
                                                    n_runs=n_runs)
        print(f"train_policy_on_ode {n_runs} runs : {time.time() - start_time}")
    