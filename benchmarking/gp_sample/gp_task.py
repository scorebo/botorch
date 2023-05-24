import sys
import os
from os.path import join, dirname, isfile, abspath
import json

import torch
from torch import Tensor
import pandas as pd

from botorch.test_functions.synthetic import SyntheticTestFunction

GP_PATH = join(dirname(__file__), 'gp_{}dim/gp_{}dim_run{}.csv')
OPT_PATH = join(dirname(__file__), 'gp_{}dim/gp_{}dim_run{}.json')
SIGMA = 1.0


class GPTestFunction(SyntheticTestFunction):

    def __init__(self,
                 dim: int,
                 seed: int = 42,
                 noise_std: float = 0,
                 scale: float = SIGMA,
                 negate: bool = True
                 ):
        self.dim = dim
        self.seed = seed
        self.scale = scale
        # with open(OPT_PATH.format(self.dim, self.dim, str(seed)), 'r') as f:
        #     opt_dict = json.load(f)
        #     self.opt = Tensor(opt_dict['X'])
        #     self.optval = Tensor([opt_dict['opt']])
            
        self._bounds = [(0, 1)] * self.dim
        super().__init__(noise_std=noise_std, negate=negate)

        data = pd.read_csv(GP_PATH.format(self.dim, self.dim, str(seed)))
        self.w = Tensor(data.iloc[:, 0:-2].to_numpy())
        self.b = Tensor(data.iloc[:, -2].to_numpy()).reshape(-1, 1)
        self.theta = Tensor(data.iloc[:, -1].to_numpy()).reshape(-1, 1)

    def evaluate_true(self, X):
        X = X.to(self.w)
        if X.ndim == 3:
            X = X.squeeze(1)
            
        return len(self.b) * torch.mean(self.theta * torch.sqrt(Tensor([2]) * self.scale / len(self.b))
                                        * torch.cos(torch.matmul(self.w, X.T) + self.b), axis=0)
