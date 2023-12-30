import contextlib
import jax
import gymnax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from functools import partial


# From equinox library
def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def make_rollout_env(environment_params):
    if "gymnax" in str(type(environment_params)):
        env, _ = gymnax.make('Pendulum-v1')

    return env

def rollout(key, model_params, model_static, env_params, steps_in_episode=None):
    """Rollout a jitted gymnax episode with lax.scan."""

    def get_action(obs, key):
      ACTIONS = jnp.array([-1, 0, 1])
      return jr.choice(key, ACTIONS, p=model(obs))

    def policy_step(state_input, tmp, env):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        action = get_action(obs, rng_net)
        next_obs, next_state, reward, done, _ = env.step(
          rng_step, state, action, env_params
        )
        carry = [next_obs, next_state, rng]
        return carry, [obs, action, reward, next_obs, done]

    rng_reset, rng_episode = jax.random.split(key, 2)
    # Init the environment
    env = make_rollout_env(env_params)
    obs, state = env.reset(rng_reset, env_params)
    # Combine model
    model = eqx.combine(model_params, model_static)

    if steps_in_episode is None:
        steps_in_episode = env_params.max_steps_in_episode

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
      partial(policy_step, env=env),
      [obs, state, rng_episode],
      (),
      steps_in_episode
    )
    
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


def rollout_parallel(*args):
    return jax.jit(
        jax.vmap(
          rollout,
          in_axes=(0, None, None, None)
        ),
        static_argnames=( 'model_static', 'env_params', 'steps_in_episode')
    )(*args)

def get_action_inx(action):
    return action + 1


def rollout(key, model_params, model_static, env_params, steps_in_episode=None):
    """Rollout a jitted gymnax episode with lax.scan."""

    def get_action(obs, key):
      ACTIONS = jnp.array([-1, 0, 1])
      return jr.choice(key, ACTIONS, p=model(obs))

    def policy_step(state_input, tmp, env):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        action = get_action(obs, rng_net)

        next_obs, next_state, reward, done, _ = env.step(
          rng_step, state, action, env_params
        )

        carry = [next_obs, next_state, rng]
        return carry, [obs, action, reward, next_obs, done]

    rng_reset, rng_episode = jax.random.split(key, 2)
    # Init the environment
    env = make_rollout_env(env_params)
    obs, state = env.reset(rng_reset, env_params)
    # Combine model
    model = eqx.combine(model_params, model_static)

    if steps_in_episode is None:
        steps_in_episode = env_params.max_steps_in_episode

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
      partial(policy_step, env=env),
      [obs, state, rng_episode],
      (),
      steps_in_episode
    )
    
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done