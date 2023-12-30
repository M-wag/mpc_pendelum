import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from helpers import dataloader
import time

def train_dynamics(
    ys,
    ts,
    ode_params,
    ode_static,
    batch_size=4,
    lr_strategy=(3e-3, ),
    steps_strategy=(500, ),
    length_strategy=(0.15,),
    print_every=100,
    *,
    key=None
):
    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, ui):
        y_pred = jax.vmap(model, in_axes=(None, 0, 0))(ti, yi[:, 0], ui)
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, y, model, opt_state):
        ui = y[:, :, -1]
        yi = y[:, :, :-1]
        loss, grads = grad_loss(model, ti, yi, ui)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    model_key, loader_key = jr.split(key, 2)

    _, length_size, _ = ys.shape

    model = eqx.combine(ode_params, ode_static)

    # Notice that up until step 500 we train on only the first 10% of each time series. 
    # This is a standard trick to avoid getting caught in a local minimum.

    losses = []
    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            losses.append(loss)
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")


    return ys, losses, model

