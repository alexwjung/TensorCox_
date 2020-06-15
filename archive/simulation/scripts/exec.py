import os 
import sys 
import torch

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append('/gpfs/nobackup/gerstung/awj/projects/TensorCox/') 
from TensorCox import loglikelihood
from TensorCox import Fisher
from TensorCox import concordance
from TensorCox import RMSE
from dataloader import dataloader
from dataloader import ToTensor
from dataloader import custom_collate
from simulation import sim_data
from simulation import _run

p_cluster = int(sys.argv[1])
c_cluster = float(sys.argv[2])
batch_cluster= int(sys.argv[3])

sample_size = []
parameters = []
censoring =[]
batch_size = []
concordance = []
RMSE_se = []
RMSE_theta = []

for n in [10000]:
    for p in [p_cluster]:
        for c in [c_cluster]:
            S_, X_, T_ = sim_data(n=n, p=p, c=c, lambda_=0.01, nu_=4)
            SE_ = torch.diagonal(torch.sqrt(torch.inverse(Fisher(torch.from_numpy(S_), torch.from_numpy(X_), torch.from_numpy(np.matmul(X_, T_)))))).detach().numpy()  
            for batchsize in [batch_cluster]:
                theta_hat, se_hat, ci = _run(S_, X_, batchsize, lr=0.01, epochs=5000)
                sample_size.append(n)
                parameters.append(p)
                censoring.append(c)
                batch_size.append(batchsize)
                concordance.append(ci)
                RMSE_theta.append(RMSE(theta_hat, T_))
                RMSE_se.append(RMSE(SE_, se_hat))

sample_size = np.squeeze(np.asarray(sample_size))[None, None]
parameters = np.squeeze(np.asarray(parameters))[None, None]
censoring = np.squeeze(np.asarray(censoring))[None, None]
batch_size = np.squeeze(np.asarray(batch_size))[None, None]
concordance = np.squeeze(np.asarray(concordance))[None, None]
RMSE_se = np.squeeze(np.asarray(RMSE_se))[None, None]
RMSE_theta = np.squeeze(np.asarray(RMSE_theta))[None, None]

data = np.concatenate((sample_size, parameters, censoring, batch_size, concordance, RMSE_se, RMSE_theta), axis=1)
data = pd.DataFrame(data)
data.columns =['sample_size', 'parameters', 'censoring', 'batch_size', 'concordance', 'RMSE_se', 'RMSE_theta']
data.to_csv('/gpfs/nobackup/gerstung/awj/projects/TensorCox/simulation/results/sim' + '_' + str(p_cluster) + '_' + str(c_cluster) + '_' + str(batch_cluster) +   '.csv', ';')









