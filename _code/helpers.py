import contextlib
import jax
import gymnax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from functools import partial


@contextlib.contextmanager
def local_gymnax_env(name):
    old_env = globals().get('env', None)  # Backup the original 'env', if it exists
    old_env_params = globals().get('env_params', None)  # Backup the original 'env', if it exists

    globals()['env'], globals()['env'] = gymnax.make(name)

    try:
        yield
    finally:
        # Clean up: Reset 'env' to its original value
        if old_env is not None:
            globals()['env'] = old_env
        else:
            del globals()['env']

        if old_env_params is not None:
            globals()['env_params'] = old_env_params
        else:
            del globals()['env_params']

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
    #   jax.debug.print(str(model(obs)))
      return jr.choice(key, ACTIONS, p=model(obs))

    def policy_step(state_input, tmp):
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

    # Get variablees
    if steps_in_episode is None:
        steps_in_episode = env_params.max_steps_in_episode

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
      policy_step,
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
