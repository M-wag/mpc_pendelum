import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
import optax

import matplotlib.pyplot as plt
from functools import partial
from helpers import rollout, rollout_parallel, get_action_inx


def get_discounted_rewards_step(carry, reward, gamma):
    """Scannable function which should be ran in reverse 

    carry: (G_init)
    reward: reward
    gamma: discount value

    """
    G_prev = carry
    G_new = reward + gamma * G_prev  
    return G_new, G_new

def get_discounted_rewards(rewards, gamma):
    G_init = 0
    _, Gs = jax.lax.scan(
        partial(get_discounted_rewards_step, gamma=gamma), 
        G_init, 
        rewards,
        reverse=True)
    return Gs


def baseline_arr(Gs):
    # Apply baseline correction
    baseline = jnp.mean(Gs)
    Gs_baseline_corrected = Gs - baseline
    Gs_standardized = (Gs_baseline_corrected - jnp.mean(Gs_baseline_corrected)) / (jnp.std(Gs_baseline_corrected) + 1e-8)

    # Return discouses accumulated rewards in proper order (start to finish)

    return Gs_baseline_corrected

def get_trajectory_gradients(model_params, model_static, Gs, obs, actions): 
    def step(carry, variables, params, static):
        @partial(jax.jit, static_argnames='static')
        def loss(params, static, obs, action, G): 
            log_prob = jnp.log(eqx.combine(params, static)(obs))
            # jax.debug.print("{x}", x=log_prob)
            action_inx = get_action_inx(action)
            log_prob_action = -G * log_prob[action_inx]
            return log_prob_action 

        # Get variables and carry
        delta = carry
        G, obs, action = variables

        #Update error terms based on gradients
        grad_delta = jax.grad(loss)(params, static, obs, action, G)
        delta = jtu.tree_map(lambda t1, t2: t1 + t2, delta, grad_delta)

        carry = delta
        return carry, G

    # Set variables
    delta = jtu.tree_map(lambda t: jnp.zeros(t.shape), model_params)
    variables = (Gs[::-1], obs[::-1], actions[::-1],)
    
    #Iterate backwards in time
    new_delta, _ =  jax.lax.scan(
        partial(step, params=model_params, static=model_static), 
        delta,
        variables
        )
    
    return new_delta

def get_trajectory_gradients_parallel(*args):
    return jax.jit(
        jax.vmap(
            get_trajectory_gradients,
            in_axes=(None, None, 0, 0, 0)
        ),
        static_argnames='model_static'
    )(*args)

@partial(jax.jit, static_argnames='static')
def loss_REINFORCE(observationss, actionss, rewardss, params, static):
    # Get discounted rewards Gs
    Gss = baseline_arr(jax.vmap(partial(get_discounted_rewards, gamma=0.99))(rewardss))
    # because equinox gradients are equinox modules batching returns 
    # a single Pytree where the leaves are batched
    # for example a weight matrix of size (a , b) -> (batch_size, a ,b)
    deltas = get_trajectory_gradients_parallel(params, static, Gss, observationss, actionss)
    delta = jtu.tree_map(lambda t: jnp.sum(t, axis=0), deltas)

    return delta, Gss
    # get delta

def visualize_trajectory(policy, env, env_params, key):
    params, static = eqx.partition(policy, eqx.is_array)
    obs, action, reward, next_obs, done = rollout(key, params, static, env_params)
    ts = jnp.linspace (0.05, env_params.dt * env_params.max_steps_in_episode, env_params.max_steps_in_episode)

    fig, ax = plt.subplots(5,1,figsize=(8,8))
    # first three plots for the system states
    ax[0].set_title('System states over time')

    for d in range(env.obs_shape[0]):
        ax[d].plot(ts, obs[:,d], color='C0', label=f'State {d}')
    ax[0].set_title(r'$\cos(\theta)$')
    ax[1].set_title(r'$\sin(\theta)$')
    ax[2].set_title(r'$\dot{\theta}$')

    ax[3].plot(ts, action, color='C1', label=f'Actions')
    # ax[3].set_ylim((env.action_space().low, env.action_space().high))
    ax[3].set_title('u(t)')
    ax[4].plot(ts, reward, color='C2', label='Rewards')
    ax[4].set_title('r(t)')

    plt.tight_layout()
    plt.show()

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def train_step(carry, key, 
               model_static, env_params, optimizer, n_batches):

    # Unpack last params and optimizer state
    params, opt_state = carry

    # Forward pass
    keys_rollout = jr.split(key, n_batches)
    obss, actionss, rewardss, _, _= rollout_parallel(keys_rollout, params, model_static, env_params)
    
    # Compute gradients
    delta, _ = loss_REINFORCE(obss, actionss, rewardss, params, model_static)
    # Update model
    updates, opt_state = optimizer.update(delta, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    carry = new_params, opt_state
    return carry, jnp.mean(jnp.sum(rewardss,axis=-1))