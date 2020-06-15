import os 
import sys 
import glob

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.style.use('seaborn-pastel')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


dir_data = '/gpfs/nobackup/gerstung/awj/projects/TensorCox/results/'


files = glob.glob(dir_data + '*')
data = pd.concat([pd.read_csv(file, sep=';') for file in files], axis=0)




estimator = 'RMSE_se'

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xticklabels(['', '5', '10', '50', '100', '500', '1000', '5000', 'full'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for c in [0.01, 0.1, 0.5]:
    for p in [10, 100, 1000]:
            _plot(ax, data, censoring=c, parameters=p, estimator=estimator)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Batch Size')
ax.set_ylabel(estimator)




def _plot(ax, data, censoring=0.1, parameters=100, estimator='concordance'):
    sub = data.loc[np.logical_and(data['censoring']==censoring, data['parameters']==parameters), :].copy()
    sub.loc[:, 'batch_size']= sub.loc[:, 'batch_size'].replace([5, 10, 50, 100, 500, 1000, 5000, 10000], [1, 2, 3, 4, 5, 6, 7, 8])
    sub.sort_values('batch_size', inplace=True)
    ax.plot(sub['batch_size'], np.round( sub[estimator], 2), label='c: ' + str(censoring) + ', p: ' +  str(parameters) )


    
    
    
