import copy
import subprocess
import numpy as np

import sys
import os
from os.path import dirname, abspath, join
from botorch.test_functions.base import BaseTestProblem
import numpy as np
import torch
from torch import Tensor
import time


class CosmoFunction(BaseTestProblem):

    def __init__(self, noise_std: float = None, negate: bool = False, seed: int = 42):
        self.seed = seed
        self.dim = 11
        self._bounds = [
            [0.01, 0.08],
            [0.01, 0.25],
            [0.01, 0.25],
            [52.5, 100],
            [2.7, 2.8],
            [0.2, 0.3],
            [2.9, 3.09],
            [1.5e-9, 2.6e-8],
            [0.72, 2.7],
            [0, 100],
            [0, 100]
        ]
        self.embedded_functions = [None, None, None]
        super().__init__(noise_std=noise_std, negate=negate)
        self.path = create_path(time.time())

    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        assert len(X) == 1
        res = cosmo(X.detach().numpy().flatten(), path=self.path)
        return Tensor([res])


def create_path(unique_index):
    experiment_path = dirname(dirname(dirname(abspath(__file__))))
    os.system(
        f'cp -r {experiment_path}/camb {experiment_path}/extra_cambs/camb_{unique_index}')
    return f'{experiment_path}/extra_cambs/camb_{unique_index}'


def cosmo(x: np.ndarray, path: str):
    PATH = path
    with open(f'{PATH}/CAMB-Feb09/params.ini', 'r') as file:
        data = file.readlines()
    data[5] = "output_root = ../lrgdr7like/models/lrgdr7model"
    data[34] = f'ombh2={x[0]}\n'
    data[35] = f'omch2={x[1]}\n'
    data[37] = f'omk={x[2]}\n'
    data[38] = f'hubble={x[3]}\n'
    data[51] = f'temp_cmb={x[4]}\n'
    data[52] = f'helium_fraction={x[5]}\n'
    data[53] = f'massless_neutrinos={x[6]}\n'
    data[69] = f'scalar_amp(1)={x[7]}\n'
    data[70] = f'scalar_spectral_index(1)={x[8]}\n'
    data[93] = f'RECFAST_fudge={x[9]}\n'
    data[94] = f'RECFAST_fudge_He={x[10]}\n'

    with open(f'{PATH}/CAMB-Feb09/params.ini', 'w') as file:
        file.writelines(data)

    try:
        out = subprocess.run(
            ["./camb", "params.ini"], cwd=f"{PATH}/CAMB-Feb09", capture_output=True, timeout=600, check=True)
    except subprocess.TimeoutExpired as e:
        print("TIMEOUT", x)
        return 300
    if out.stderr or len(out.stdout) <= 3 or out.stdout[:3] != b"Age":
        print(out.stderr, out.stdout, x)
        return 300
    res = subprocess.run(["./getlrgdr7like"],
                         cwd=f"{PATH}/lrgdr7like", capture_output=True)
    try:
        return float(res.stdout.strip())
    except:
        print(res.stdout)
        print(res.stderr)
        print(x)
        # return 300
