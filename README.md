# score-sde

## Install
```
git clone --recurse-submodules https://github.com/oxcsml/score-sde.git
cd score-sde
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -e .
# pip install -r requirements.txt
```

## Run experiments

### S^2 toy
`python main.py experiment=s2_toy`