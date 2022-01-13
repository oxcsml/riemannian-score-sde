import os
import hydra
from score_sde.utils.cfg import *

@hydra.main(config_path="config", config_name="main")
def main(cfg):
    os.environ['GEOMSTATS_BACKEND'] = 'jax'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    from run import run

    return run(cfg)

if __name__ == "__main__":
    main()