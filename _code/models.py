
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import diffrax

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_size, output_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, us, inx):
        u = us[inx]
        y = jnp.concatenate([y, jnp.expand_dims(u, -1)])
        return self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, input_size, output_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(input_size, output_size, width_size, depth, key=key)

    def __call__(self, ts, y0, us):
        def ode_func(t, y, args):
            us, ts = args
            inx = jnp.argmax(ts == t)  
            return self.func(t, y, us, inx)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            args=(us, ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

class NeuralNetwork(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_size, output_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.relu,
            key=key,
        )
    
    def __call__(self, y):
        return jnn.softmax(self.mlp(y))
