# import the necessary modules
import sys
import os
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 



x = np.linspace(0., 100., 1000)
y = np.sin(x/10)
plt.scatter(x, y)

x_h = np.arange(0, 125, 25)
x_h = torch.from_numpy(x_h)
x = torch.from_numpy(x)
y = torch.from_numpy(y)

scale = 20
y = ((-(x[:, None] - x_h)**2)/scale**2).exp()
for _ in range(10):
    theta = torch.normal(0, 1, (5, 1), dtype=torch.float64, requires_grad=True)
    a = torch.mm(y, theta)
    plt.plot(x, a.detach())
plt.show()
plt.close()


def spline(x, x_h, scale,  dtype=torch.float64):
    delta = x_h[1] - x_h[0]
    
    u1 = (x - x_h[0]) / delta
    b1 = 1/6 * u1**3
    b1 = b1 * ((x > x_h[0]) * (x <= x_h[1])).type(dtype)
    
    u2 = (x - x_h[1]) / delta
    b2 = 1/6 * (1 + 3*u2 + 3*u2**2 - 3*u2**3)
    b2 = b2 * ((x > x_h[1]) * (x <= x_h[2])).type(dtype)
    
    u3= (x - x_h[2]) / delta
    b3 = 1/6 * (4 - 6*u3**2 + 3*3**3)
    b3 = b3 * ((x > x_h[2]) * (x <= x_h[3])).type(dtype)
    
    u4 = (x - x_h[3]) / delta
    b4 = 1/6 *(1 - 3*u4 + 3*u4**2 - u4**3)
    b4 = b4 * ((x > x_h[3])).type(dtype)
    return(torch.cat((torch.ones((x.shape[0], 1)).type(dtype), b1[:, None], b2[:, None], b3[:, None], b4[:, None]), axis=1))
    


def spline(x, x_h, dtype=torch.float64):
    delta = x_h[1] - x_h[0]
    
    u1 = (x - x_h[0]) / delta
    b1 = 1/6 * u1**3
    b1 = b1 * ((x > x_h[0]) * (x <= x_h[1])).type(dtype)
    
    u2 = (x - x_h[1]) / delta
    b2 = 1/6 * (1 + 3*u2 + 3*u2**2 - 3*u2**3)
    b2 = b2 * ((x > x_h[1]) * (x <= x_h[2])).type(dtype)
    
    u3= (x - x_h[2]) / delta
    b3 = 1/6 * (4 - 6*u3**2 + 3*3**3)
    b3 = b3 * ((x > x_h[2]) * (x <= x_h[3])).type(dtype)
    
    u4 = (x - x_h[3]) / delta
    b4 = 1/6 *(1 - 3*u4 + 3*u4**2 - u4**3)
    b4 = b4 * ((x > x_h[3])).type(dtype)
    return(torch.cat((torch.ones((x.shape[0], 1)).type(dtype), x[:, None], b1[:, None], b2[:, None], b3[:, None], b4[:, None]), axis=1))
    

    
theta = torch.normal(0, 1, (6, 1), dtype=torch.float64, requires_grad=True)

eta = 0.00
lr = 0.1
optimizer = torch.optim.Adam([theta], lr=lr)

for _ in range(10000):
    optimizer.zero_grad()
    ypred = torch.mm(spline(x, x_h), theta)[:, 0]
    loss = torch.mean((y-ypred)**2)
    loss.backward()
    optimizer.step()

plt.scatter(x, y, color='red')
plt.plot(x, ypred.detach().numpy(), color='blue')


loss = torch.nn.MSELoss()

output = loss(input, target)


ypred.shape













































