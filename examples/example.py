# import the necessary modules
import sys
import os
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader

# set root directory
# path to the folder
dir_root = '/gpfs/nobackup/gerstung/awj/projects/TensorCox/'
os.chdir(dir_root)

# appends the path to the COX script 
sys.path.append(dir_root + 'TensorCox/')

# import COX model
from TensorCox import loglikelihood
from TensorCox import Fisher
from metrics import concordance
from metrics import RMSE
from dataloader import CSV_Dataset
from dataloader import ToTensor
from dataloader import custom_collate

torch.manual_seed(7)
np.random.seed(7)

# define data loader
ddata = CSV_Dataset(path_data='data/lung.csv', batch_size=200, transform=ToTensor())
data_pipeline = DataLoader(ddata, batch_size=1, num_workers=1, collate_fn=custom_collate)

# optimizer
parameters = 7
theta = torch.normal(0, 0.01, (parameters, 1), dtype=torch.float64, requires_grad=True)
eta = 0.00
lr = 0.001
optimizer = torch.optim.SGD([theta], lr=lr)

for _ in range(1000):
    for batch_idx, (surv, X) in enumerate(data_pipeline):
        if torch.sum(surv[:, -1]) > 0:
            X = (X - torch.mean(X, axis=0))/torch.std(X, axis=0)
            optimizer.zero_grad()
            linpred = torch.mm(X, theta)
            logL = -loglikelihood(surv, linpred) 
            logL.backward()
            optimizer.step()
        else:
            next
print(theta)     

# Fisher information
A = np.zeros((7,7))
A = torch.from_numpy(A)
for _ in range(1):
    for batch_idx, (surv, X) in enumerate(data_pipeline):
        X = (X - torch.mean(X, axis=0))/torch.std(X, axis=0)
        linpred = torch.mm(X, theta)
        A += Fisher(surv, X, linpred)
print(torch.diagonal(torch.sqrt(torch.inverse(A))))

# load all data in for concordance 
ddata = CSV_Dataset(path_data='data/lung.csv', batch_size=1000, transform=ToTensor())
data_pipeline = DataLoader(ddata, batch_size=1, num_workers=1, collate_fn=custom_collate)
for _ in range(1):
    for batch_idx, (surv, X) in enumerate(data_pipeline):
        X = (X - torch.mean(X, axis=0))/torch.std(X, axis=0)
        linpred = torch.mm(X, theta)
        break
print(concordance(surv.numpy(), linpred.detach().numpy()))
