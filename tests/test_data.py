# import for pytest
if __name__ != "__main__":
    from .._code.data import get_data

# when running directly
if __name__ == "__main__":
    import SETUP_PATH
    from _code.data import get_data

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.random import PRNGKey
import optimistix as optx

import pytest
from functools import partial


@partial(vmap, in_axes=(0, 0, None))

def solution_de(y0, args, ts):
    
    def _solution_de(args, t):
        y_new = solve(t, args)

        return args, y_new

    def solve(t, args): 
        a, b = args
        return jnp.exp(a * t) + b * t


    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    sol = optx.root_find(solve, solver, y0, args)
    ts += sol.value

    print(sol.value.val)

    _, solutions = lax.scan(_solution_de, args, ts)

    return solutions 
    
def de(t, y, args):
    a, b = args
    return a * jnp.exp(a * y) + b

@pytest.mark.skip(reason="root finder not working when exponent is enabled")
def test_get_data_matches_de_solution():
    key = PRNGKey(0)
    args = jnp.array([0.1, 2])

    n_runs = 1
    y0s = jax.random.uniform(key, shape=(n_runs,), minval=2.0, maxval=3.0)
    arggs = jnp.tile(args, (n_runs, 1))

    n_datapoints = 5
    ts = jnp.linspace(0, 0.01, n_datapoints)

    # Get data for analytical solution and numerically solved solution
    data_numeric = get_data(de, y0s, arggs, ts, n_datapoints, key)
    data_analytical = solution_de(y0s, arggs, ts)

    print(data_analytical)
    print(data_numeric)
    # Compute the absolute difference
    difference = jnp.abs(data_analytical - data_numeric)

    # Normalize by the magnitude of data_numeric, adding a small epsilon to avoid division by zero
    normalized_difference = difference / (jnp.abs(data_numeric) + 1e-12)


    # Define the tolerance (eight orders of magnitude smaller)
    tolerance = 1e-3

    # Assert that the normalized difference is below the tolerance for all elements
    assert jnp.all(normalized_difference < tolerance), \
        f"The difference between numeric and analytical solutions is not within the expected tolerance of {tolerance}."

    


if __name__ == '__main__':
    test_get_data_matches_de_solution()