import os, sys
import pandas as pd
import numpy as np
import pickle

def unpickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

save_dir = './kent'
fns = os.listdir(save_dir)

rows = []
for fn in fns:
    fp = os.path.join(save_dir, fn)
    out = unpickle(fp)
    dataset = "_".join(fp.split('_')[:-4]).split('/')[-1]
    row = {**out['inputs'], **{'train_loglik': out['train_loglik'],
                              'test_loglik': out['test_loglik'],
                              'eval_loglik': out['eval_loglik'],
                              'dataset': dataset}}
    rows.append(row)
           



df = pd.DataFrame.from_records(rows)

datasets = df.dataset.unique()

print(datasets)

df['mean_loglik'] = df.test_loglik
df['std_loglik'] = df.test_loglik
df['count'] = df.seed

# hack in case eval group too small and has nan - should not do anything
df['eval_loglik'] = np.where(pd.isna(df.eval_loglik), df.train_loglik, df.eval_loglik )


results = []
for data_tag in datasets:
    print(data_tag)
    rview = df.dataset == data_tag
    res = df[rview].groupby(['dataset',
                             'n_components', 'iterations']) \
             .agg({'train_loglik': 'mean',
                   'eval_loglik': 'mean',
                   'mean_loglik': 'mean',
                   'std_loglik': 'std',
                   'count': 'count'})  \
             .reset_index() 
    
    res = res[res['count']>1] \
          .sort_values(by='eval_loglik', ascending=False) \
          .iloc[0:1,:]
    results.append(res)


print(pd.concat(results, axis=0))