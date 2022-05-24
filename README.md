# score-sde

## Install
This repo requires a modified version of [geomstats](https://github.com/geomstats/geomstats) that adds jax functionality, and a number of other modifications. This can be found [here](https://github.com/oxcsml/geomstats.git ) on the branch `jax_backend`.

Simple install instructions are:
```
git clone https://github.com/oxcsml/score-sde.git
cd score-sde
git clone --single-branch --branch jax_backend https://github.com/oxcsml/geomstats.git 
virtualenv -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_exps.txt
pip install -e geomstats
pip install -e .
```

- `requirements.txt` contains the core requirements for running the code in the `score_sde` and `riemmanian_score_sde` packages. NOTE: you ma need to alter the jax versions here to match your setup.
- `requirements_exps.txt` contains extra dependencies needed for running our experiments, and using the `run.py` file provided for training / testing models. 
- `requirements_slurm.txt` contains extra dependencies for using the job scheduling functionality of hydra.
- `requirements_dev.txt` contains some handy development packages.

## Run experiments

### S^2 toy
`python main.py experiment=s2_toy`

### SO(3) toy
`python main.py experiment=so3 dataset=wrapped`