# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Gaussian Process Regression models with fully Bayesian inference.

Fully Bayesian models use Bayesian inference over model hyperparameters, such
as lengthscales and noise variance, learning a posterior distribution for the
hyperparameters using the No-U-Turn-Sampler (NUTS). This is followed by
sampling a small set of hyperparameters (often ~16) from the posterior
that we will use for model predictions and for computing acquisition function
values. By contrast, our “standard” models (e.g.
`SingleTaskGP`) learn only a single best value for each hyperparameter using
MAP. The fully Bayesian method generally results in a better and more
well-calibrated model, but is more computationally intensive. For a full
description, see [Eriksson2021saasbo].

We use a lightweight PyTorch implementation of a Matern-5/2 kernel as there are
some performance issues with running NUTS on top of standard GPyTorch models.
The resulting hyperparameter samples are loaded into a batched GPyTorch model
after fitting.

References:

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.
"""


import math
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple, Callable, Union

import pyro
import numpy as np
import torch
from torch.distributions import Kumaraswamy
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.sampling import MCSampler
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior, MCMC_DIM
from botorch.models.transforms.input import Warp
from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.kernels.kernel import dist, Kernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from botorch.models.gpytorch import ModelListGPyTorchModel
import gpytorch
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.linear_mean import PolynomialMean
from gpytorch.means.mean import Mean
from gpytorch.means import ZeroMean
from gpytorch.models.exact_gp import ExactGP
from pyro.ops.integrator import register_exception_handler
from torch import Tensor
from torch.linalg import inv, solve_triangular, cholesky
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP


MIN_INFERRED_NOISE_LEVEL = 1e-6

_sqrt5 = math.sqrt(5)


def _handle_torch_linalg(exception: Exception) -> bool:
    return type(exception) == torch.linalg.LinAlgError


def _handle_valerr_in_dist_init(exception: Exception) -> bool:
    if not type(exception) == ValueError:
        return False
    return "satisfy the constraint PositiveDefinite()" in str(exception)


register_exception_handler("torch_linalg", _handle_torch_linalg)
register_exception_handler("valerr_in_dist_init", _handle_valerr_in_dist_init)


def compute_mean(X: Tensor, const: int, poly: Tensor) -> Tensor:
    degrees = torch.arange(1, poly.shape[-1] + 1)
    x_poly = torch.pow(X.unsqueeze(-1), degrees)
    return const + torch.sum(x_poly * poly, dim=[-1, -2])


# TODO exponential kernel
def sqexp_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Squared exponential kernel."""

    dist = compute_dists(X=X, lengthscale=lengthscale)
    exp_component = torch.exp(-torch.pow(dist, 2) / 2)
    return exp_component


def matern52_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Matern-5/2 kernel."""
    dist = compute_dists(X=X, lengthscale=lengthscale)
    sqrt5_dist = _sqrt5 * dist
    return sqrt5_dist.add(1 + 5 / 3 * (dist**2)) * torch.exp(-sqrt5_dist)


def split_matern52_kernel(X: Tensor, lengthscale: Tensor, groups: int, splits: Tensor) -> Tensor:
    """Matern-5/2 kernel."""
    K = torch.zeros((len(X), len(X)))
    for group in range(groups):
        split = (splits == group).flatten()
        X_split = X[..., split]
        ls_split = lengthscale[..., split]

        dist = compute_dists(X=X_split, lengthscale=ls_split)

        sqrt5_dist = _sqrt5 * dist
        K_comp = sqrt5_dist.add(1 + 5 / 3 * (dist**2)) * torch.exp(-sqrt5_dist)
        K = K + K_comp
    return K


def compute_dists(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Compute kernel distances."""
    scaled_X = X / lengthscale
    return dist(scaled_X, scaled_X, x1_eq_x2=True)


def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape `new_value` to match `target`."""
    return new_value.detach().clone().view(target.shape).to(target)


class PyroModel:
    r"""
    Base class for a Pyro model; used to assist in learning hyperparameters.

    This class and its subclasses are not a standard BoTorch models; instead
    the subclasses are used as inputs to a `SaasFullyBayesianSingleTaskGP`,
    which should then have its hyperparameters fit with
    `fit_fully_bayesian_model_nuts`. (By default, its subclass `SaasPyroModel`
    is used).  A `PyroModel`’s `sample` method should specify lightweight
    PyTorch functionality, which will be used for fast model fitting with NUTS.
    The utility of `PyroModel` is in enabling fast fitting with NUTS, since we
    would otherwise need to use GPyTorch, which is computationally infeasible
    in combination with Pyro.

    :meta private:
    """

    def set_warm_start_state(self, warm_start_state):
        self.warm_start_state = warm_start_state

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        """Set the training data.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
        """
        self.custom_fit = False
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar

    @abstractmethod
    def sample(self) -> None:
        r"""Sample from the model."""
        pass  # pragma: no cover

    @abstractmethod
    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Post-process the final MCMC samples."""
        pass  # pragma: no cover

    @abstractmethod
    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        pass  # pragma: no cover


class SaasPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        tausq = pyro.sample(
            "kernel_tausq",
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq",
            pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq", tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale",
            inv_length_sq.rsqrt(),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        inv_length_sq = (
            mcmc_samples["kernel_tausq"].unsqueeze(-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = inv_length_sq.rsqrt()
        # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
        # into the final model.
        del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class WarpingPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        eps = torch.finfo(tkwargs["dtype"]).eps
        outputscale = self.sample_outputscale(**tkwargs)

        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        c0, c1 = self.sample_input_warping(dim=self.ard_num_dims, **tkwargs)
        # unnormalize X from [0, 1] to [eps, 1-eps]
        X = self.train_X
        X = (X * (1 - 2 * eps) + eps).clamp(eps, 1 - eps)
        X_tf = 1 - torch.pow((1 - torch.pow(X, c1)), c0)

        K = matern52_kernel(X=X_tf, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def transform(self, X: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        r"""
        Transforms the input using the inverse CDF of the Kumaraswamy distribution.
        """
        k_dist = Kumaraswamy(
            concentration1=beta,
            concentration0=alpha,
        )
        return k_dist.cdf(torch.clamp(X, 0.0, 1.0))

    def sample_input_warping(
        self,
        dim: int,
        mean: float = 0.0,
        variance: float = 0.1,
        **tkwargs: Any,
    ) -> Tuple[Tensor, Tensor]:

        c0 = pyro.sample(
            "c0",
            # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
            pyro.distributions.LogNormal(
                torch.tensor([mean] * dim, **tkwargs),
                torch.tensor([variance**0.5] * dim, **tkwargs),
            ),
        )
        c1 = pyro.sample(
            "c1",
            # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
            pyro.distributions.LogNormal(
                torch.tensor([mean] * dim, **tkwargs),
                torch.tensor([variance**0.5] * dim, **tkwargs),
            ),
        )
        return c0, c1

    def sample_outputscale(self, mean: float = 0.0, variance: float = 1, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 1, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 1, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        # by default, all inputs are warped
        warping_indices = list(range(self.train_X.shape[-1]))
        input_warping = Warp(
            warping_indices, batch_shape=mcmc_samples["c0"].shape[:-1])
        input_warping._set_concentration(0, mcmc_samples["c0"])
        input_warping._set_concentration(1, mcmc_samples["c1"])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )

        return mean_module, covar_module, likelihood, input_warping


class WarpingWithoutOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        eps = torch.finfo(tkwargs["dtype"]).eps

        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        c0, c1 = self.sample_input_warping(dim=self.ard_num_dims, **tkwargs)
        # unnormalize X from [0, 1] to [eps, 1-eps]
        X = self.train_X
        X = (X * (1 - 2 * eps) + eps).clamp(eps, 1 - eps)
        X_tf = 1 - torch.pow((1 - torch.pow(X, c1)), c0)

        K = matern52_kernel(X=X_tf, lengthscale=lengthscale)
        K = K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def transform(self, X: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        r"""
        Transforms the input using the inverse CDF of the Kumaraswamy distribution.
        """
        k_dist = Kumaraswamy(
            concentration1=beta,
            concentration0=alpha,
        )
        return k_dist.cdf(torch.clamp(X, 0.0, 1.0))

    def sample_input_warping(
        self,
        dim: int,
        mean: float = 0.0,
        variance: float = 0.75,
        **tkwargs: Any,
    ) -> Tuple[Tensor, Tensor]:

        c0 = pyro.sample(
            "c0",
            # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
            pyro.distributions.LogNormal(
                torch.tensor([mean] * dim, **tkwargs),
                torch.tensor([variance**0.5] * dim, **tkwargs),
            ),
        )
        c1 = pyro.sample(
            "c1",
            # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
            pyro.distributions.LogNormal(
                torch.tensor([mean] * dim, **tkwargs),
                torch.tensor([variance**0.5] * dim, **tkwargs),
            ),
        )
        return c0, c1

    def sample_noise(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        # by default, all inputs are warped
        warping_indices = list(range(self.train_X.shape[-1]))
        input_warping = Warp(
            warping_indices, batch_shape=mcmc_samples["c0"].shape[:-1])
        input_warping._set_concentration(0, mcmc_samples["c0"])
        input_warping._set_concentration(1, mcmc_samples["c1"])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=torch.ones_like(mcmc_samples["noise"]),
        )

        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )

        return mean_module, covar_module, likelihood, input_warping


class BayesOptWithoutOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        eps = torch.finfo(tkwargs["dtype"]).eps

        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)

        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=torch.ones_like(mcmc_samples["noise"]),
        )

        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )

        return mean_module, covar_module, likelihood


class SaasFullyBayesianSingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and thatcdcd
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`. The SAAS model [Eriksson2021saasbo]_
    with a Matern-5/2 kernel is used by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        pyro_model: Optional[PyroModel] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,
        disable_progbar: bool = False,
        num_groups: int = 0,
        hyperparameters: dict = None,
        warm_start_state: Optional[Any] = None,

    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            pyro_model: Optional `PyroModel`, defaults to `SaasPyroModel`.
        """
        if isinstance(train_Yvar, float):
            train_Yvar = torch.full_like(train_Y, train_Yvar)
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )
        num_mcmc_samples = num_samples // thinning
        if pyro_model is None:
            pyro_model = SaasPyroModel()

        pyro_model.set_inputs(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self.pyro_model = pyro_model
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)
        self._num_outputs = train_Y.shape[-1]
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        X_tf, Y_tf, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        super().__init__(
            train_inputs=X_tf, train_targets=Y_tf, likelihood=GaussianLikelihood()
        )
        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        self.pyro_model.set_warm_start_state(warm_start_state=warm_start_state)
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.num_samples = num_samples
        self.warmup = warmup
        self.thinning = thinning
        self.disable_progbar = disable_progbar
        if num_groups > 0 and hasattr(self.pyro_model, 'num_groups'):
            self.pyro_model.set_num_groups(groups=num_groups)
            self.pyro_model.set_hyperparameters(**hyperparameters)

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return len(self.covar_module.outputscale)

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that `SaasFullyBayesianSingleTaskGP` does not support batching
        over input data at this point."""
        return torch.Size([self.num_mcmc_samples])

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Custom logic for loading the state dict.

        The standard approach of calling `load_state_dict` currently doesn't play well
        with the `SaasFullyBayesianSingleTaskGP` since the `mean_module`, `covar_module`
        and `likelihood` aren't initialized until the model has been fitted. The reason
        for this is that we don't know the number of MCMC samples until NUTS is called.
        Given the state dict, we can initialize a new model with some dummy samples and
        then load the state dict into this model. This currently only works for a
        `SaasPyroModel` and supporting more Pyro models likely requires moving the model
        construction logic into the Pyro model itself.
        """

        if not isinstance(self.pyro_model, SaasPyroModel):
            raise NotImplementedError("load_state_dict only works for SaasPyroModel")
        raw_mean = state_dict["mean_module.raw_constant"]
        num_mcmc_samples = len(raw_mean)
        dim = self.pyro_model.train_X.shape[-1]
        tkwargs = {"device": raw_mean.device, "dtype": raw_mean.dtype}
        # Load some dummy samples
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "lengthscale": torch.ones(num_mcmc_samples, dim, **tkwargs),
            "outputscale": torch.ones(num_mcmc_samples, **tkwargs),
        }
        if self.pyro_model.train_Yvar is None:
            mcmc_samples["noise"] = torch.ones(num_mcmc_samples, **tkwargs)
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        predict_per_model: bool = False,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        # X_batch = X.unsqueeze(0).repeat(self.num_batch, 1, 1, 1)
        if predict_per_model:
            posterior = super().posterior(
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        else:
            posterior = super().posterior(
                X=X.unsqueeze(MCMC_DIM),
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        if self.pyro_model.custom_fit:
            from botorch.fit import fit_fully_bayesian_model_custom
            fit_fully_bayesian_model_custom(
                model=self,
                warmup_steps=self.warmup,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=self.disable_progbar
            )
        else:
            from botorch.fit import fit_fully_bayesian_model_nuts
            fit_fully_bayesian_model_nuts(
                model=self,
                warmup_steps=self.warmup,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=self.disable_progbar
            )

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Union[bool, Tensor] = True,
        **kwargs: Any,
    ):
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X` (if `observation_noise=True`,
        this includes observation noise taken as the mean across the observation
        noise in the training data. If `observation_noise` is a Tensor, use
        it directly as the observation noise to add).
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: If True, include the mean across the observation
                noise in the training data as observation noise in the posterior
                from which the samples are drawn. If a Tensor, use it directly
                as the specified measurement noise.

        Returns:
            The constructed fantasy model.
        """
        propagate_grads = kwargs.pop("propagate_grads", False)

        with fantasize_flag():
            with settings.propagate_grads(propagate_grads):
                post_X = self.posterior(
                    X, observation_noise=observation_noise, **kwargs
                )
            X = X.unsqueeze(MCMC_DIM).repeat(1, post_X.shape()[1], 1, 1)

            Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
            # Use the mean of the previous noise values (TODO: be smarter here).
            # noise should be batch_shape x q x m when X is batch_shape x q x d, and
            # Y_fantasized is num_fantasies x batch_shape x q x m.
            noise_shape = Y_fantasized.shape[1:]
            noise = self.likelihood.noise.unsqueeze(-1).expand(noise_shape)
            return self.condition_on_observations(
                X=self.transform_inputs(X), Y=Y_fantasized, noise=noise
            )

    def get_warm_start_state(self):
        return self.pyro_model.get_warm_start_state()


class WarpingFullyBayesianSingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and thatcdcd
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`. The SAAS model [Eriksson2021saasbo]_
    with a Matern-5/2 kernel is used by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        pyro_model: Optional[PyroModel] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,
        disable_progbar: bool = False
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            pyro_model: Optional `PyroModel`, defaults to `SaasPyroModel`.
        """
        num_mcmc_samples = num_samples // thinning
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)

        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)

        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )
        self._num_outputs = train_Y.shape[-1]
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        X_tf, Y_tf, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        super().__init__(
            train_inputs=X_tf, train_targets=Y_tf, likelihood=GaussianLikelihood()
        )

        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        if pyro_model is None:
            pyro_model = SaasPyroModel()

        pyro_model.set_inputs(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self.pyro_model = pyro_model
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.num_samples = num_samples
        self.warmup = warmup
        self.thinning = thinning
        self.disable_progbar = disable_progbar
        # self.num_batch = num_mcmc_samples

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return len(self.covar_module.outputscale)

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that `SaasFullyBayesianSingleTaskGP` does not support batching
        over input data at this point."""
        return torch.Size([self.num_mcmc_samples])

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
            self.input_transform
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Custom logic for loading the state dict.

        The standard approach of calling `load_state_dict` currently doesn't play well
        with the `SaasFullyBayesianSingleTaskGP` since the `mean_module`, `covar_module`
        and `likelihood` aren't initialized until the model has been fitted. The reason
        for this is that we don't know the number of MCMC samples until NUTS is called.
        Given the state dict, we can initialize a new model with some dummy samples and
        then load the state dict into this model. This currently only works for a
        `SaasPyroModel` and supporting more Pyro models likely requires moving the model
        construction logic into the Pyro model itself.
        """

        if not isinstance(self.pyro_model, SaasPyroModel):
            raise NotImplementedError("load_state_dict only works for SaasPyroModel")
        raw_mean = state_dict["mean_module.raw_constant"]
        num_mcmc_samples = len(raw_mean)
        dim = self.pyro_model.train_X.shape[-1]
        tkwargs = {"device": raw_mean.device, "dtype": raw_mean.dtype}
        # Load some dummy samples
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "lengthscale": torch.ones(num_mcmc_samples, dim, **tkwargs),
            "outputscale": torch.ones(num_mcmc_samples, **tkwargs),
        }
        if self.pyro_model.train_Yvar is None:
            mcmc_samples["noise"] = torch.ones(num_mcmc_samples, **tkwargs)
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        predict_per_model: bool = False,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        # X_batch = X.unsqueeze(0).repeat(self.num_batch, 1, 1, 1)

        if predict_per_model:
            posterior = super().posterior(
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )

        else:
            posterior = super().posterior(
                X=X.unsqueeze(MCMC_DIM),
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        from botorch.fit import fit_fully_bayesian_model_nuts
        fit_fully_bayesian_model_nuts(
            model=self,
            warmup_steps=self.warmup,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=self.disable_progbar
        )


class ActiveLearningPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, mean: float = 0.0, variance: float = 1e-4, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return torch.tensor([1.0], **tkwargs)

    def sample_noise(self, mean: float = -1.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return torch.tensor([0.0], **tkwargs)

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.tensor(torch.ones(dim) * mean, **tkwargs),
                torch.tensor(torch.ones(dim) * variance ** 0.5, **tkwargs),
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ZeroMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=torch.ones_like(mcmc_samples["noise"]),
        )

        return mean_module, covar_module, likelihood


class InitPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 0.0, variance: float = 1, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 1, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 1, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class ActiveLearningWithOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class QuadraticPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        CENTERING_CONST = 0.5
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        const = self.sample_const(**tkwargs)
        poly = self.sample_poly(dim=self.ard_num_dims, **tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        mean = compute_mean(self.train_X - CENTERING_CONST, const, poly)

        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(1.1, **tkwargs),
                    torch.tensor(0.05, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.Gamma(
                (torch.ones(dim) * 3.0).to(**tkwargs),
                (torch.ones(dim) * 6.0).to(**tkwargs),
            ),
        )
        return lengthscale

    def sample_const(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "const",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_poly(self, dim: int, degree: int = 2, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "poly",
            pyro.distributions.Normal(
                torch.zeros((dim, degree)).to(**tkwargs),
                1 * torch.ones((dim, degree)).to(**tkwargs),
            ),
        )

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        degree = mcmc_samples["poly"].shape[-1]
        has_bias = "const" in mcmc_samples.keys()
        batch_shape = torch.Size([num_mcmc_samples])
        mean_module = PolynomialMean(
            bias=has_bias, input_size=self.ard_num_dims, degree=degree, batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        if has_bias:
            mean_module.bias.data = reshape_and_detach(
                target=mean_module.bias.data,
                new_value=mcmc_samples["const"],
            )
        mean_module.weights.data = reshape_and_detach(
            target=mean_module.weights.data,
            new_value=mcmc_samples["poly"],
        )

        return mean_module, covar_module, likelihood


class ActiveLearningWithoutMeanPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return torch.tensor([0.0], **tkwargs)

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ZeroMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )

        return mean_module, covar_module, likelihood


class BadActiveLearningWithOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 4.0, variance: float = 4.0, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 4.0, variance: float = 4.0, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 4.0, variance: float = 4.0, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class WideActiveLearningWithOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 0.0, variance: float = 6.0, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 6.0, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 6.0, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class SingleTaskPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(1.1, **tkwargs),
                    torch.tensor(0.05, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return torch.tensor([0.0], **tkwargs)

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.Gamma(
                (torch.ones(dim) * 3.0).to(**tkwargs),
                (torch.ones(dim) * 6.0).to(**tkwargs),
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ZeroMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )

        return mean_module, covar_module, likelihood


class SingleTaskMeanPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(1.1, **tkwargs),
                    torch.tensor(0.05, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.Gamma(
                (torch.ones(dim) * 3.0).to(**tkwargs),
                (torch.ones(dim) * 6.0).to(**tkwargs),
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class PyroFullyBayesianSingleTaskGP(SingleTaskGP):
    r"""A fully Bayesian single-task GP model where the sampler is passed in.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`.

    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        pyro_model: PyroModel,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,
        disable_progbar: bool = False

    ):

        self.pyro_model = pyro_model
        self.pyro_model.set_inputs(train_X, train_Y)
        num_mcmc_samples = num_mcmc_samples = int(num_samples / thinning)
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)
        super().__init__(train_X, train_Y, outcome_transform=outcome_transform,
                         input_transform=input_transform)
        self.covar_module = None
        self.num_samples = num_samples
        self.warmup = warmup
        self.thinning = thinning

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        return super().forward(X)

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        from botorch.fit import fit_fully_bayesian_model_nuts
        fit_fully_bayesian_model_nuts(
            model=self,
            warmup_steps=self.warmup,
            num_samples=self.num_samples,
            thinning=self.thinning,
        )

    def load_mcmc_samples(self, mcmc_samples) -> None:
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

        # The input data should come in with zero mean, so this will just be zero
        # Could probably make this into a ZeroMean, too

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X.unsqueeze(MCMC_DIM),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior


def normal_tensor_likelihood(hp_tensor, train_X, train_Y, gp_kernel='matern'):
    outputscale = hp_tensor[0]
    noise = hp_tensor[1]
    lengthscales = hp_tensor[2:]
    return normal_log_likelihood(train_X, train_Y, outputscale, noise, lengthscales, gp_kernel=gp_kernel)


def normal_tensor_likelihood_noscale(hp_tensor, train_X, train_Y, gp_kernel='matern'):
    outputscale = torch.ones_like(hp_tensor[0])
    noise = hp_tensor[0]
    lengthscales = hp_tensor[1:]
    return normal_log_likelihood(train_X, train_Y, outputscale, noise, lengthscales, gp_kernel=gp_kernel)


def warped_tensor_mean_likelihood(hp_tensor, train_X, train_Y, gp_kernel='matern'):
    dim = train_X.shape[-1]
    mean = hp_tensor[0]
    outputscale = hp_tensor[1]
    noise = hp_tensor[2]
    lengthscales = hp_tensor[3:3 + dim]
    alphas = hp_tensor[3 + dim:3 + dim * 2]
    betas = hp_tensor[3 + 2 * dim:3 + dim * 3]

    return normal_log_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        mean=mean,
        outputscale=outputscale,
        noise=noise,
        lengthscales=lengthscales,
        alphas=alphas,
        betas=betas,
        gp_kernel=gp_kernel
    )


def transform(X: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    r"""
    Transforms the input using the inverse CDF of the Kumaraswamy distribution.
    """
    k_dist = Kumaraswamy(
        concentration1=beta,
        concentration0=alpha,
    )
    return k_dist.cdf(torch.clamp(X, 0.0, 1.0))


def normal_log_likelihood(train_X, train_Y, outputscale, noise, lengthscales, mean=0, alphas=None, betas=None, gp_kernel='matern'):
    if (alphas is not None) and (betas is not None):
        train_X = transform(train_X, alpha=alphas, beta=betas)
    if gp_kernel == "matern":
        k_noiseless = matern52_kernel(X=train_X, lengthscale=lengthscales)
    elif gp_kernel == "rbf":
        k_noiseless = sqexp_kernel(X=train_X, lengthscale=lengthscales)
    else:
        raise ValueError('Not a valid kernel, choose matern or rbf.')

    train_Y = train_Y - mean

    k = outputscale * k_noiseless + noise * \
        torch.eye(train_X.shape[0], dtype=train_X.dtype, device=train_X.device)

    L_cholesky = cholesky(k)
    Lf_y = solve_triangular(L_cholesky, train_Y, upper=False)
    Lf_yy = solve_triangular(L_cholesky.T, Lf_y, upper=True)

    const = 0.5 * len(train_Y) * torch.log(Tensor([2 * torch.pi]))
    logdet = torch.sum(torch.log(torch.diag(L_cholesky)))
    # ensure this actually returns a scalar
    logprobs = 0.5 * torch.matmul(train_Y.T, Lf_yy)
    return -(const + logdet + logprobs)


class EllipticalSliceSampler:

    def __init__(self, prior, lnpdf, num_samples, pdf_params=(), warmup=0, thinning=1):
        r"""
        RETRIEVED FROM Wesley Maddox's github (and adapted ever so slightly): 
        https://github.com/wjmaddox/pytorch_ess/blob/main/pytorch_ess/elliptical_slice.py
        Implementation of elliptical slice sampling (Murray, Adams, & Mckay, 2010).
        The current implementation assumes that every parameter is log normally distributed.
        """

        # TODO allow for labeling of each dimension so that things don't have to come in the
        # order outputscale, noise, lengthscale
        self.n = prior.mean.nelement()
        self.prior = prior
        self.lnpdf = lnpdf
        self.num_samples = num_samples + warmup
        self.pdf_params = pdf_params
        self.warmup = warmup
        self.thinning = thinning
        self.f_priors = (prior.rsample(torch.Size(
            [self.num_samples])) - prior.mean.unsqueeze(0))
        self.num_mcmc_samples = int(num_samples / thinning)

    def untransform_sample(self, sample):
        return torch.exp(sample + self.prior.mean)

    def get_samples(self):
        raw_samples, likelihood = self.run()
        raw_samples = raw_samples
        thinned_samples = raw_samples[self.warmup::self.thinning, :]
        return self.untransform_sample(thinned_samples)

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.f_sampled = torch.zeros(
            self.num_samples, self.n, device=self.f_priors.device, dtype=self.f_priors.dtype)
        self.ell = torch.zeros(
            self.num_samples, 1, device=self.f_priors.device, dtype=self.f_priors.dtype)

        f_cur = torch.zeros_like(self.prior.mean)
        for ii in range(self.num_samples):
            if ii == 0:
                ell_cur = self.lnpdf(self.untransform_sample(f_cur), *self.pdf_params)
            else:
                # Retrieves the previous best - it is a markov chain =)
                # could probably do with some warmup and thinning for diversity then as well
                f_cur = self.f_sampled[ii - 1, :]
                ell_cur = self.ell[ii - 1, 0]

            next_f_prior = self.f_priors[ii, :]

            self.f_sampled[ii, :], self.ell[ii] = self.elliptical_slice(f_cur, next_f_prior,
                                                                        cur_lnpdf=ell_cur, pdf_params=self.pdf_params)

        return self.f_sampled, self.ell

    def elliptical_slice(
        self,
        initial_theta: torch.Tensor,
        prior: torch.Tensor,
        pdf_params: Tuple = (),
        cur_lnpdf: torch.Tensor = None,
        angle_range: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = len(initial_theta)
        if cur_lnpdf is None:
            cur_lnpdf = self.lnpdf(self.untransform_sample(initial_theta), *pdf_params)

        ## FORCING THE RIGHT PRIOR TO BE GIVEN ##
        # Set up the ellipse and the slice threshold
        if len(prior.shape) == 1:  # prior = prior sample
            nu = prior
        else:  # prior = cholesky decomp
            if not prior.shape[0] == D or not prior.shape[1] == D:
                raise IOError(
                    "Prior must be given by a D-element sample or DxD chol(Sigma)")
            nu = prior.transpose(-1, -2).matmul(torch.randn(
                prior.shape[:-2], 1, device=prior.device, dtype=prior.dtype))

        hh = torch.rand(1).log() + cur_lnpdf

        # Set up a bracket of angles and pick a first proposal.
        # "phi = (theta'-theta)" is a change in angle.
        if angle_range is None or angle_range == 0.:
            # Bracket whole ellipse with both edges at first proposed point
            phi = torch.rand(1) * 2. * math.pi
            phi_min = phi - 2. * math.pi
            phi_max = phi
        else:
            # Randomly center bracket on current point
            phi_min = -angle_range * torch.rand(1)
            phi_max = phi_min + angle_range
            phi = torch.rand(1) * (phi_max - phi_min) + phi_min

        # Slice sampling loop
        while True:
            # Compute xx for proposed angle difference and check if it's on the slice
            xx_prop = initial_theta * math.cos(phi) + nu * math.sin(phi)

            cur_lnpdf = self.lnpdf(self.untransform_sample(xx_prop), *pdf_params)
            if cur_lnpdf > hh:
                # New point is on slice, ** EXIT LOOP **
                break
            # Shrink slice to rejected point
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise RuntimeError(
                    'BUG DETECTED: Shrunk to current position and still not acceptable.')
            # Propose new angle difference
            phi = torch.rand(1) * (phi_max - phi_min) + phi_min

        return (xx_prop, cur_lnpdf)

    def load_mcmc_samples(self):
        samples = self.get_samples()
        samples_dict = {}
        ndim = int((samples.shape[1] - 3) / 3)
        samples_dict['mean'] = samples[: , 0]
        samples_dict['outputscale'] = samples[: , 1]
        samples_dict['noise'] = samples[: , 2]
        samples_dict['lengthscale'] = samples[: , 3:3 + ndim]
        samples_dict['alpha'] = samples[: , 3 + ndim:3 + 2 * ndim]
        samples_dict['beta'] = samples[: , 3 + 2 * ndim:3 + 3 * ndim]
        return samples_dict


class AdditivePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None, groups: int = 0
    ):

        self.custom_fit = False
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        #self.custom_fit = True
        self.ard_num_dims = self.train_X.shape[-1]
        self.set_num_groups(groups=groups)

    def set_num_groups(self, groups: int = 0):
        if groups == 0:
            self.num_groups = math.ceil(math.sqrt(self.train_X.shape[-1]))
        else:
            self.num_groups = groups

    def set_hyperparameters(
        self,
        noise: Tensor,
        outputscale: Tensor,
        lengthscale: Tensor,
        mean: Optional[Tensor] = None,
        active_groups: Optional[Tensor] = None
    ) -> None:

        self.active_groups = active_groups

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        #mean = self.sample_mean(**tkwargs)
        #noise = self.sample_noise(**tkwargs)
        mean = torch.Tensor([0]).to(**tkwargs)
        noise = torch.Tensor([0.25]).to(**tkwargs)
        splits = self.active_groups
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = split_matern52_kernel(
            X=self.train_X, lengthscale=lengthscale, splits=splits, groups=self.num_groups)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.Gamma(
                (torch.ones(dim) * 3.0).to(**tkwargs),
                (torch.ones(dim) * 6.0).to(**tkwargs),
            ),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        self.active_groups = self.active_groups.unsqueeze(0).repeat(num_mcmc_samples, 1)
        dim_groups = self.active_groups
        dim_range = torch.arange(self.ard_num_dims)
        mean_modules = []
        covar_modules = []
        likelihoods = []
        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)

        active_dims = torch.cat(
            [dim_groups.unsqueeze(-1) == i for i in range(self.num_groups)], -1)

        # create the additive matern kernels. Each kernel recieves all the lengthscales (and then
        # just separates out the relevant dimensions internally). A specific batch component of
        # one additive kernel may have no active dimensions (since there is a max number of groups,
        # but not a minimum) in which case there is no correlation whatsoever.
        for group_idx in range(self.num_groups):
            # the kernel for the first group
            if group_idx == 0:
                matern_kernels = MaternKernel(
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=batch_shape,
                    active_dims=active_dims[..., group_idx]
                )
                matern_kernels.lengthscale = reshape_and_detach(
                    target=matern_kernels.lengthscale,
                    new_value=mcmc_samples["lengthscale"],
                )
                # the kernels for the subsequent groups
            else:
                group_kernel = MaternKernel(
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=batch_shape,
                    active_dims=active_dims[..., group_idx]
                )
                group_kernel.lengthscale = reshape_and_detach(
                    target=group_kernel.lengthscale,
                    new_value=mcmc_samples["lengthscale"],
                )
                matern_kernels = group_kernel + matern_kernels

            covar_module = ScaleKernel(
                base_kernel=matern_kernels,
                batch_shape=batch_shape,
            ).to(**tkwargs)

        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=0.25
                * torch.ones_like(mcmc_samples["outputscale"]
                                  ).clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=torch.zeros_like(mcmc_samples["outputscale"]),
        )

        return mean_module, covar_module, likelihood


class AdditiveFullyBayesianSingleTaskGP(ModelListGP):
    r"""A fully Bayesian single-task GP model with additive structure.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        *models,
        train_X: Tensor = None,
        train_Y: Tensor = None,
        train_Yvar: Optional[Tensor] = None,
        pyro_model: Optional[PyroModel] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,
        disable_progbar: bool = False,
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            pyro_model: Optional `PyroModel`, defaults to `SaasPyroModel`.
        """
        if len(models) == 0:
            self.train_X_temp = train_X
            self.train_Y_temp = train_Y
            # initializes a placeholder model
            if train_Yvar is not None:
                self.model_class = FixedNoiseGP
                super().__init__(self.model_class(self.train_X_temp, self.train_Y_temp, train_Yvar))
            else:
                self.model_class = SingleTaskGP
                super().__init__(self.model_class(self.train_X_temp, self.train_Y_temp))

            self.pyro_model = pyro_model
            self.num_samples = num_samples
            self.warmup = warmup
            self.thinning = thinning
            self.disable_progbar = disable_progbar
            pyro_model.set_inputs(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar
            )
        else:
            super().__init__(*models)

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        (
            mean_modules,
            covar_modules,
            likelihoods,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        models = [
            self.model_class(
                train_X=self.train_X_temp,
                train_Y=self.train_Y_temp,
                mean_module=mean_modules[i],
                covar_module=covar_modules[i],
                likelihood=likelihoods[i],
            )
            for i in range(len(covar_modules))
        ]
        super().__init__(*models)

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        from botorch.fit import fit_fully_bayesian_model_nuts
        fit_fully_bayesian_model_nuts(
            model=self,
            warmup_steps=self.warmup,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=self.disable_progbar
        )


class FixedAdditiveSamplingModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        groups: int = 0,
    ):
        self.custom_fit = False
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.custom_fit = True
        self.ard_num_dims = self.train_X.shape[-1]
        self.set_num_groups(groups=groups)

    def set_num_groups(self, groups: int = 0):
        if groups == 0:
            self.num_groups = math.ceil(math.sqrt(self.train_X.shape[-1]))
        else:
            self.num_groups = groups

    def set_hyperparameters(
        self,
        noise: Tensor,
        outputscale: Tensor,
        lengthscale: Tensor,
        mean: Optional[Tensor] = None,
        active_groups: Optional[Tensor] = None
    ) -> None:

        self.lengthscale = lengthscale
        self.noise = noise
        self.outputscale = outputscale
        if mean is not None:
            self.mean = mean
        else:
            self.mean = Tensor([0])

        self.sample_groups = active_groups is None
        if not self.sample_groups:
            self.active_groups = active_groups

    def evaluate_likelihood(self, splits, lengthscale, noise, outputscale, mean) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        # Tensor([0.5] * self.ard_num_dims).to(**tkwargs)
        lengthscale = lengthscale.to(**tkwargs)
        noise = noise.to(**tkwargs)  # Tensor([0.5]).to(**tkwargs)
        outputscale = outputscale.to(**tkwargs)  # Tensor([1]).to(**tkwargs)
        mean = mean.to(**tkwargs)

        K = split_matern52_kernel(
            X=self.train_X, lengthscale=lengthscale, splits=splits, groups=self.num_groups)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        try:
            L_cholesky = cholesky(K)
        except:
            return -torch.inf
        Lf_y = solve_triangular(L_cholesky, self.train_Y - mean, upper=False)
        Lf_yy = solve_triangular(L_cholesky.T, Lf_y, upper=True)

        const = 0.5 * len(self.train_Y) * torch.log(Tensor([2 * torch.pi]))
        logdet = torch.sum(torch.log(torch.diag(L_cholesky)))
        # ensure this actually returns a scalar
        logprobs = 0.5 * torch.matmul(self.train_Y.T - mean, Lf_yy)
        return -(const + logdet + logprobs)

    def sample_split(self, splits: Tensor, num_swaps: int = 2, **tkwargs: Any) -> Tensor:
        if splits is None:
            dist = torch.distributions.Uniform(
                low=torch.zeros(self.ard_num_dims), high=torch.ones(self.ard_num_dims) * self.num_groups)
            return torch.floor(dist.rsample(sample_shape=torch.Size([1]))).to(torch.long)
        else:
            splits = splits.detach().numpy()
            swap_indices = np.random.choice(
                len(splits.T), size=num_swaps, replace=False)
            new_values = np.random.choice(self.num_groups)
            splits[:, swap_indices] = new_values

            return Tensor(splits)

    def run(
        self,
        num_chains: int = 3,
        warmup_steps: int = 256,
        num_samples: int = 256,
        thinning: int = 16,
        disable_progbar: bool = False
    ):
        num_chains = 1
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        splits = None
        all_models = []
        all_params = []
        if self.warm_start_state is None:
            splits = self.sample_split(splits, **tkwargs)
            last_split = splits
        else:
            splits = self.warm_start_state.clone()
            last_split = self.warm_start_state.clone()

        old_model, old_likelihood = self._evaluate_likelihood(splits)

        for sample_idx in range(warmup_steps + num_samples):
            splits = self.sample_split(splits, **tkwargs)
            model, likelihood = self._evaluate_likelihood(splits)
            log_unif = torch.log(torch.rand(num_chains))
            if likelihood > (old_likelihood + log_unif):
                old_likelihood = likelihood
                old_model = model
                #old_params = params
                last_split = splits
            all_models.append(old_model)
            # all_params.append(old_params)

        self.last_split = last_split
        # print(all_params)
        return {'group': all_models[warmup_steps:]}

    def get_warm_start_state(self):
        return self.last_split

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        models = mcmc_samples["group"]
        num_mcmc_samples = torch.Size([len(models)])
        state_dict = self._separate_models(models)

        for group_idx in range(self.num_groups):
            # the kernel for the first group
            if group_idx == 0:
                matern_kernels = RBFKernel(
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=num_mcmc_samples,
                    active_dims=torch.ones((len(models), self.ard_num_dims))
                )
            else:
                group_kernel = RBFKernel(
                    ard_num_dims=self.ard_num_dims,
                    batch_shape=num_mcmc_samples,
                    active_dims=torch.ones((len(models), self.ard_num_dims))
                )
                matern_kernels = group_kernel + matern_kernels

            covar_module = ScaleKernel(matern_kernels).to(**tkwargs)

        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples + (len(self.train_Yvar), )),
                batch_shape=num_mcmc_samples,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=num_mcmc_samples    ,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
        covar_module.load_state_dict(filter_args(state_dict, 'covar_module'))
        mean_module = ConstantMean(batch_shape=num_mcmc_samples).to(**tkwargs)
        mean_module.load_state_dict(filter_args(state_dict, 'mean_module'))
        if self.train_Yvar is None:
            likelihood.noise = state_dict['likelihood.noise_covar.raw_noise']
        return mean_module, covar_module, likelihood

    def _evaluate_likelihood(self, active_groups: Tensor):
        active_dims = torch.cat(
            [active_groups == i for i in range(self.num_groups)], 0)
        train_X = self.train_X
        train_Y = self.train_Y

        for group_idx in range(self.num_groups):
            # the kernel for the first group
            if group_idx == 0:
                matern_kernels = RBFKernel(
                    ard_num_dims=self.ard_num_dims,
                    active_dims=active_dims[group_idx, ...],
                    # lengthscale_constraint=ls_constraints
                )
                # the kernels for the subsequent groups
            else:
                group_kernel = RBFKernel(
                    ard_num_dims=self.ard_num_dims,
                    active_dims=active_dims[group_idx, ...],
                    # lengthscale_constraint=ls_constraints
                )
                matern_kernels = group_kernel + matern_kernels

        covar_module = ScaleKernel(matern_kernels)

        from botorch.fit import fit_gpytorch_mll
        from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
        from gpytorch.mlls import ExactMarginalLogLikelihood
        import warnings
        if self.train_Yvar is None:
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                                 covar_module=covar_module)
        else:
            model = FixedNoiseGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=self.train_Yvar,
                covar_module=covar_module,
            )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        lengthscale = torch.zeros((1, train_X.shape[-1])).to(self.train_X)
        for kern in covar_module.base_kernel.kernels:
            lengthscale[:, kern.active_dims] = kern.lengthscale[:, kern.active_dims]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.eval()
            output = model(self.train_X)

        model.train()
        try:
            fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_scipy,
                             optimizer_kwargs={'options': {}})
        except:
            warnings.warn(
                f'Failed fitting with parameters {active_groups}. Defaulting.')
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.train()
            output = model(self.train_X)
        log_likelihood = mll(output, self.train_Y.flatten()) * len(self.train_Y)
        noise = model.likelihood.noise
        lengthscale = torch.zeros((1, train_X.shape[-1])).to(self.train_X)
        for kern in covar_module.base_kernel.kernels:
            lengthscale[:, kern.active_dims] = kern.lengthscale[:, kern.active_dims]
        #   log_likelihood_2 = self.evaluate_likelihood(
        #       active_groups, lengthscale=lengthscale, outputscale=outputscale, noise=noise, mean=mean)
        return model.state_dict(), log_likelihood  # log_likelihood.unsqueeze(0)

    def _separate_models(self, state_dicts):
        total_state_dict = {}
        for i, sd in enumerate(state_dicts):
            for param_name, param_val in sd.items():
                if i == 0:
                    total_state_dict[param_name] = param_val.unsqueeze(0)
                else:
                    total_state_dict[param_name] = torch.cat(
                        (total_state_dict[param_name], param_val.unsqueeze(0)), dim=0)

        return total_state_dict


def filter_args(state_dict, module_name, drop=None):
    new_dict = {}
    for key, val in state_dict.items():
        if module_name in key:
            if drop is None or all(d not in key for d in drop):
                new_key = key.replace(f'{module_name}.', '')
                new_dict[new_key] = val

    return new_dict
