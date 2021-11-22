from collections import namedtuple


TrainState = namedtuple(
    "TrainState",
    [
        "opt_state",
        "model_state",
        "step",
        "params",
        "ema_rate",
        "params_ema",
        "rng",
    ],
)
