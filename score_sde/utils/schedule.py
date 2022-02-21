import jax.numpy as jnp


def loglinear_schedule(
    init_value,
    end_value,
    decay_steps,
):

    log_init = jnp.log(init_value)
    log_end = jnp.log(end_value)

    def schedule(count):
        t = count / decay_steps
        return jnp.exp(log_init + t * (log_end - log_init))

    return schedule
