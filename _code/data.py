import diffrax
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import lax
from jax import vmap
from functools import partial
import matplotlib.pyplot as plt



def ode_rollout(f, y0s, args, ts, key):
    """
    Performs a vector-mapped data rollout for a differential equation.

    Parameters:
        f (Callable): The differential equation function. It should define the system's dynamics.
        y0s (array):An array of initial conditions for the differential equation. 
                    Each element in the array corresponds to a separate rollout. 
                    If all rollouts should have the same initial condition, ensure that the 
                    elements in this vector are identical.
        args (set): Additional arguments required by the differential equation `f`.
        ts: The time axis over which to simulate the differential equation. This could be a 
            range, list, or array of time points.
        key (PRNGKey):
    """

    @partial(vmap, in_axes=(None, 0, None, None, 0))
    def _ode_rollout(f, y0, args, ts, key):
        solver = diffrax.Tsit5()
        dt0 = ts[1] - ts[0]
        saveat = diffrax.SaveAt(ts=ts)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, args, saveat=saveat
        )
        return sol.ys

    # y0s = set_rank_2D(y0s)
    
    n_runs = y0s.shape[0]
    keys = jr.split(key, n_runs)
    ys = _ode_rollout(f, y0s, args, ts, keys)

    return ys

def set_rank_2D(array):
    rank = len(array.shape)

    if rank == 2:
        return array
    elif rank == 1:
        return jnp.expand_dims(array, axis=0)
    else:
        raise ValueError("Input array must be either 1D or 2D")

    

    
def rollout(rng_input, policy_params, env, env_params, steps_in_episode):
    def get_action(params, observation, key):
        action =  jr.choice(key, jnp.array([-1, 0, 1]))
        action_idx = action + 1 
        return action, action_idx

    def policy_step(step_input, temp):
        """Step function compatible with lax.scan"""
        obs, state, policy_params, rng = step_input

        # Split key 
        rng, rng_action, rng_step = jr.split(rng, 3)
        # Take an action
        action, action_idx = get_action(policy_params, obs, rng_action)
        # Take a step
        next_obs, next_state, reward, done, _ = env.step(
                rng_step, state, action, env_params)

        carry = [next_obs, next_state, policy_params, rng]

        return carry, [obs, state, action, reward, next_obs, done]

    rng_reset, rng_episode = jr.split(rng_input)
    # Reset environment

    if hasattr(env, 'reset'):
        env.reset()
        
    obs, state = env.reset(rng_reset, env_params)

    # Scan over episode step loop
    _, step_scan = jax.lax.scan(
        policy_step,
        init = [obs, state, policy_params, rng_episode],
        xs = None,
        length = steps_in_episode)

    return step_scan


def visualize_trajectory(key, params, env, env_params, steps_in_episode=200):
    obs, state, action, reward, next_obs, done = rollout(key, params, env, env_params, 
                                                   steps_in_episode=steps_in_episode)


    ts = jnp.linspace(0, action.size, action.size)

    fig, ax = plt.subplots(5,1,figsize=(8,8))
    # first three plots for the system states
    ax[0].set_title('System states over time')

    for d in range(env.obs_shape[0]):
        ax[d].plot(ts, obs[:,d], color='C0', label=f'State {d}')
    ax[0].set_title(r'$\cos(\theta)$')
    ax[1].set_title(r'$\sin(\theta)$')
    ax[2].set_title(r'$\dot{\theta}$')
    
    ax[3].plot(ts, action, color='C1', label=f'Actions')
    ax[3].set_title('u(t)')
    ax[4].plot(ts, reward, color='C2', label='Rewards')
    ax[4].set_title('r(t)')

    plt.tight_layout()
    plt.show()
