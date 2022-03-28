# score-sde

## Install
```
<!-- git clone --recurse-submodules https://github.com/oxcsml/score-sde.git -->
git clone https://github.com/oxcsml/score-sde.git
cd score-sde
git clone --single-branch --branch jax_backend https://github.com/oxcsml/geomstats.git 
virtualenv -p python3.9 venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e geomstats
```

## Run experiments

### S^2 toy
`python main.py experiment=s2_toy`

### SO(3) toy
`python main.py experiment=so3 dataset=wrapped`