# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial 


if __name__ == '__main__':
    from _code.reinforce import baseline_arr, get_discounted_rewards_step, get_discounted_rewards

    ts = jnp.linspace(0, 20, 100)
    rewards = jnp.ones_like(ts)
    gamma= 0.9
    G_init = 0

    _, Gs_manual = jax.lax.scan(
        partial(get_discounted_rewards_step, gamma=gamma), 
        G_init, 
        rewards,
        reverse=True)
    
    Gs = get_discounted_rewards(rewards, gamma)


    plt.plot(rewards)
    plt.plot(Gs)
    plt.plot(Gs_manual)

    assert jnp.all(Gs == Gs_manual),\
    f"Manual scanning and get_discounted_rewards should return same output"
else:
    import pytest
    from ._code.reinforce import baseline_arr, get_discounted_rewards

    # Should be generalized for all timepoints
    # And for different Cs
    @pytest.mark.parametrize("gamma", [(0.0), (0.5), (1.0)])
    def test_G0_calculation(gamma):
        c = 1.0
        ts = jnp.linspace(0, 20, 100)
        steps = jnp.arange(0, ts.size)
        rewards = jnp.ones_like(ts) * c

        G0_calculated = jnp.sum(c * gamma ** steps)

        _, Gs = jax.lax.scan(
            partial(get_discounted_rewards, gamma=gamma), 
            0, 
            rewards,
            reverse=True)
        
        # Check if the first element of Gs matches G0_calculated
        G0_from_scan = Gs[0]
        assert jnp.isclose(G0_calculated, G0_from_scan),\
        f"Expected {G0_calculated}, got {G0_from_scan}, for c={c}"



