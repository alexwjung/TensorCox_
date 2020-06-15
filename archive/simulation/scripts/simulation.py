import sys
import torch
import numpy as np 
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append('/gpfs/nobackup/gerstung/awj/projects/TensorCox/') 
from TensorCox import loglikelihood
from TensorCox import Fisher
from TensorCox import concordance
from TensorCox import RMSE
from dataloader import dataloader
from dataloader import ToTensor
from dataloader import custom_collate

def sim_data(n, p,  c=0.5, lambda_=0.01, nu_=4):
    X = np.random.normal(0, 1, (n, p))/np.sqrt(p)
    theta =  np.random.normal(0, 1, (p, 1))
    linpred_ = np.matmul(X, theta)
    times = lambda_**-1 * (-np.log(np.random.uniform(0, 1, linpred_.shape[0])) * np.exp(-linpred_[:, 0]))**(1/nu_)
    events = np.random.binomial(1, 1-c, n)
    times[-(events-1)] = np.random.uniform(0, times[-(events-1)])   
    return(np.concatenate((np.zeros((n, 1)), times[:, None], events[:, None]), axis=1), X, theta)

def _run(S_, X_, batch_size, lr=0.01, epochs=2500):
    
    d = dataloader(S_, X_, transform=ToTensor())
    data = DataLoader(d, batch_size=batch_size, num_workers=1, collate_fn=custom_collate)
    # optimizer
    parameters = X_.shape[1]
    theta = torch.normal(0, 0.01, (parameters, 1), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(epochs):
        for batch_idx, (surv, X) in enumerate(data):
            if torch.sum(surv[:, -1]) > 0:
                optimizer.zero_grad()
                linpred = torch.mm(X, theta)
                logL = -loglikelihood(surv, linpred) 
                logL.backward()
                optimizer.step()
            else:
                next  
            
    # Fisher information
    with torch.no_grad(): 
        F = np.zeros((X_.shape[1], X_.shape[1]))
        F = torch.from_numpy(F)
        for _ in range(1):
            for batch_idx, (surv, X) in enumerate(data):
                linpred = torch.mm(X, theta)
                F += Fisher(surv, X, linpred) 
    se_hat = torch.diagonal(torch.sqrt(torch.inverse(F))).detach().numpy()     
    theta_hat = theta.detach().numpy()
    ci = concordance(S_, np.matmul(X_, theta_hat))
    return(theta_hat, se_hat, ci)