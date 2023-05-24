import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import dirname, join, abspath

import torch
from torch import Size, Tensor
from torch.distributions import Gamma
from botorch.utils.sampling import draw_sobol_normal_samples


N_WEIGHTS = 2048
NU = 2.5

def generate_gp_rng(dim, n_weights, n_gps, ls, matern=True):
    lengthscale = torch.Tensor(ls)
    
    weights = draw_sobol_normal_samples(n=n_weights, d=dim) / lengthscale
    if matern:
        weights = Gamma(NU, NU).rsample((n_weights, 1)).rsqrt() * weights
    b = torch.rand(size=torch.Size([n_weights])) * math.pi
    # one theta vector represents one sample from the posterior
    theta = torch.randn(size=Size([n_weights, n_gps]))

    data = {}
    for dim_idx in range(dim):
        data[f'w_{dim_idx}'] = weights[:, dim_idx].detach().numpy()
    data['b'] = b.detach().numpy()

    for run_idx in range(n_gps):
        data[f'theta_{run_idx}'] = theta[:, run_idx].detach().numpy()
    
    return data


if __name__ == '__main__':
    n_gps, dims = sys.argv[1:3]
    matern = sys.argv[3]
    ls = sys.argv[4:]
    # The number of unique GP sample tasks to generate
    n_gps = int(n_gps)

    # The dimensionality of the GP
    dims = int(dims)
    ls_text = '_'.join(ls)
    
    # The lengthscale of the GP
    lengthscales = np.array(ls).astype(np.float32)
    
    data = generate_gp_rng(dims, N_WEIGHTS, n_gps, lengthscales, matern=bool(matern))

    all_data = pd.DataFrame(data)
    dp = join((dirname(abspath(__file__))), f'gp_{dims}dim')
        
    try:
        os.system(f'rm {dp}/LENGTHSCALE*')
    except:
        pass
    
    os.system(f'touch {dp}/LENGTHSCALE{ls_text}')
    for run_idx in range(n_gps):
        fp = join(dp, f'gp_{dims}dim_run{run_idx}.csv')
        
        try:
            os.makedirs(dp)
        except FileExistsError:
            pass
            
        all_data.iloc[:, np.append(
            np.arange(dims + 1), run_idx + dims + 1)].to_csv(fp, index=False)
