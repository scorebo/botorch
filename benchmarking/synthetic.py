from typing import List
import math
import torch
import numpy as np
from torch import Tensor

from botorch.test_functions.synthetic import SyntheticTestFunction, Branin


class Gramacy1(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 1
        self._bounds = [(0.5, 2.5)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (torch.sin(10 * math.pi * X) / (2 * X) + torch.pow(X - 1, 4)).flatten()


class Gramacy2(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 2
        self._bounds = [(-2.0, 6.0), (-2.0, 6.0)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (X[:, 0] * torch.exp(-torch.pow(X[:, 0], 2) - torch.pow(X[:, 1], 2))).flatten()


class Higdon(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 1
        self._bounds = [(0.0, 20.0)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        x = X.detach().numpy()
        output = np.piecewise(x,
                              [x < 10, x >= 10],
                              [lambda xx: np.sin(np.pi * xx / 5) + 0.2 * np.cos(4 * np.pi * xx / 5),
                                  lambda xx: xx / 10 - 1])
        return Tensor(output).to(X.device).flatten()


class Ishigami(SyntheticTestFunction):
    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 3
        self._bounds = [(-3.141527, 3.141527)] * self.dim
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        a = 7
        b = 0.1
        return (torch.sin(X[:, 0]) + a * torch.pow(torch.sin(X[:, 1]), 2)
                + b * torch.pow(X[:, 2], 4) * torch.sin(X[:, 0])).flatten()


class Embedded(SyntheticTestFunction):

    def __init__(
        self,
        function: SyntheticTestFunction,
        dim=2,
        noise_std: float = 0.0,
        negate: bool = False,
        bounds: Tensor = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        assert dim >= function.dim, 'The effective function dimensionality is larger than the embedding dimension.'
        self.dim = dim
        self._bounds = [(0.0, math.pi) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer(
            "i", torch.tensor(tuple(range(1, self.dim + 1)), dtype=torch.float)
        )
        self.embedded_function = function

    def evaluate_true(self, X: Tensor) -> Tensor:
        embedded_X = X[:, 0: self.embedded_function.dim]
        return self.embedded_function.evaluate_true(embedded_X)


class AdditiveEmbedded(SyntheticTestFunction):

    def __init__(
        self,
        functions: List[SyntheticTestFunction],
        dim=2,
        noise_std: float = 0.0,
        negate: bool = False,
        bounds: Tensor = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        self.dim_per_task = [f.dim for f in functions]
        _bounds = torch.cat([f.bounds for f in functions], dim=1)
        remaining_dims = dim - _bounds.shape[1]
        
        if remaining_dims < 0:
            raise ValueError('Too many functions for too few dimensions.'
                             f'Got {self.bounds.shape[1]} total function dims for {dim} ambient dimensions.')
        elif remaining_dims > 0:
            _bounds = torch.cat((_bounds, Tensor([[0, 1]] * remaining_dims).T), dim=1)
        self._bounds = _bounds.T
        super().__init__(noise_std=noise_std, negate=negate)
        self.embedded_functions = functions

    def evaluate_true(self, X: Tensor) -> Tensor:
        dim_count = 0
        summed_f = torch.zeros(X.shape[0])
        for dim, fun in zip(self.dim_per_task, self.embedded_functions):
            summed_f = summed_f + \
                fun.evaluate_true(X[..., dim_count:dim_count + dim])
            dim_count += dim
        return summed_f
