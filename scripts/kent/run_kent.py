import jax
import numpy as np
import os, sys
import hydra
import pickle



from scripts.kent.kent_model import KentMixture, to_cartesian_coords
from score_sde.datasets.split import random_split



def run_kent(X, initializations, n_components, iterations=100, seed=0):

    inputs = {'initializations': initializations,
              'n_components' : n_components,
              'iterations': iterations,
              'seed': seed
             }
    
    random_state = np.random.RandomState(seed=seed)
    rng = jax.random.PRNGKey(seed)
    
    datasets= random_split(X, [0.8,0.1,0.1], rng)
    train_X, eval_X, test_X = [d.dataset[d.indices] for d in datasets]
    
    klf = KentMixture(n_components=n_components, n_iter=iterations, 
                      n_init=initializations, random_state=random_state)
    klf.fit(train_X)
    
    kmm_lpr, kmm_responsibilities = klf.score_samples(train_X)
    train_log_lik = np.mean(kmm_lpr)
    
    kmm_lpr, kmm_responsibilities = klf.score_samples(eval_X)
    eval_log_lik = np.mean(kmm_lpr)
    
    kmm_lpr, kmm_responsibilities = klf.score_samples(test_X)
    test_log_lik = np.mean(kmm_lpr)
    
    
    return {'train_loglik' : train_log_lik, 
            'test_loglik' : test_log_lik, 
            'eval_loglik' : eval_log_lik, 
            'params': klf.get_params(), 
            'inputs': inputs, 
            'converged': klf.converged_}
    
def get_loglik(params, X):
    klf = KentMixture()
    klf.set_params(params)
    kmm_lpr, kmm_responsibilities = klf.score_samples(X)
    log_lik = np.mean(kmm_lpr)
    return log_lik
    
@hydra.main(config_path="./", config_name="config")
def main(cfg):

    # load and format data
    folder = cfg.data_folder
    file = os.path.join(folder, cfg.dataset +'.csv')

    X = np.genfromtxt(file, delimiter=",", skip_header=2)
    N = X.shape[0]
    intrinsic_data = (
        np.pi * (X / 180.0) + np.array([np.pi / 2, np.pi])[None, :]
    )
    X = to_cartesian_coords(intrinsic_data)

    output = run_kent(X, cfg.initializations, cfg.n_components, cfg.iterations, cfg.seed)

    output_dir = cfg.output_dir
    file_name = f'{cfg.dataset}_{cfg.seed}_{cfg.n_components}_{cfg.initializations}_{cfg.iterations}.pkl'
    fp = os.path.join(output_dir, file_name)
    with open(fp, 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':

    main()
    
