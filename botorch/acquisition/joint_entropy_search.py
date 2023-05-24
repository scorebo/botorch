#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for joint entropy search (JES). The code utilizes the
implementation designed for the multi-objective batch setting.

"""

from __future__ import annotations

from typing import Any, Optional
import warnings
from math import log, pi
import torch.distributions as dist

import torch
from torch import Tensor

from torch.distributions import Normal

from botorch.models.utils import check_no_nans
from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import is_fully_bayesian
from botorch.exceptions.warnings import BotorchTensorDimensionWarning

MCMC_DIM = -3  # Only relevant if you do Fully Bayesian GPs.
ESTIMATION_TYPES = ["MC", "LB"]
MC_ADD_TERM = 0.5 * (1 + log(2 * pi))

# The CDF query cannot be strictly zero in the division
# and this clamping helps assure that it is always positive.
CLAMP_LB = torch.finfo(torch.float32).eps
FULLY_BAYESIAN_ERROR_MSG = """JES is not yet available with Fully Bayesian GPs. Track the issue,
which regards conditioning on a number of optima on a collecion of models,
in detail at https://github.com/pytorch/botorch/issues/1680.
"""

"""
References
.. [Hvarfner2022joint]
    C. Hvarfner, F. Hutter, L. Nardi,
    Joint Entropy Search for Maximally-informed Bayesian Optimization. 
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.

.. [Tu2022joint]
    B. Tu, A. Gandy, N. Kantas, B. Shafei,
    Joint Entropy Search for Multi-objective Bayesian Optimization. 
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.
"""


class qJointEntropySearch(AcquisitionFunction, MCSamplerMixin):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point `X` and the optimal input-output pair.

    See [Tu2022]_ for a discussion on the estimation procedure.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "LB",
        maximize: bool = True,
        num_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            optimal_inputs: A `num_samples x d`-dim tensor containing the sampled
                optimal inputs of dimension `d`. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            optimal_outputs: A `num_samples x 1`-dim Tensor containing the optimal
                set of objectives of dimension `1`.
            condition_noiseless: Whether to condition on noiseless optimal observations
                f* [Hvarfner et. al.]or noisy optimal observations y* [Tu et. al,].
                These are sampled identically, so this only controls the fashion in
                which the GP is reshaped as a result of conditioning on the optimum.
            posterior_transform: A PosteriorTransform (optional).
            estimation_type: estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB") or "Monte Carlo" ("MC").
                Lower Bound is recommended due to the relatively high variance
                of the MC estimator.
            maximize: If true, we consider a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
        """
        super().__init__(model=model)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        # To enable fully bayesian GP conditioning, we need to unsqueeze
        # to get num_optima x num_gps unique GPs

        # inputs come as num_optima_per_model x (num_models) x d
        # but we want it four-dimensional in the Fully bayesian case,
        # and three-dimensional otherwise.
        self.optimal_inputs = optimal_inputs.unsqueeze(-2)
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)
        self.posterior_transform = posterior_transform
        self.maximize = maximize

        # The optima (can be maxima, can be minima) come in as the largest
        # values if we optimize, or the smallest (likely substantially negative)
        # if we minimize. Inside the acquisition function, however, we always
        # want to consider MAX-values. As such, we need to flip them if
        # we want to minimize.
        if not self.maximize:
            optimal_outputs = -optimal_outputs
        self.num_samples = optimal_inputs.shape[0]
        self.condition_noiseless = condition_noiseless
        self.initial_model = model
        tkwargs = {"dtype": optimal_outputs.dtype, "device": optimal_outputs.device}

        # Here, the optimal inputs have shapes num_optima x [num_models if FB] x 1 x D
        # and the optimal outputs have shapes num_optima x [num_models if FB] x 1 x 1
        # The third dimension equaling 1 is required to get one optimum per model,
        # which raises a BotorchTensorDimensionWarning.
        #if isinstance(model, SaasFullyBayesianSingleTaskGP):
        #    raise NotImplementedError(FULLY_BAYESIAN_ERROR_MSG)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with fantasize_flag():
                with settings.propagate_grads(False):
                    post_ps = self.initial_model.posterior(
                        self.model.train_inputs[0], observation_noise=False
                    )
                    sample_idx = 0

                # This equates to the JES version proposed by Hvarfner et. al.
                if self.condition_noiseless:
                    opt_noise = torch.full_like(
                        self.optimal_outputs, MIN_INFERRED_NOISE_LEVEL)
                    # conditional (batch) model of shape (num_models) x num_optima_per_model
                    self.conditional_model = self.initial_model.condition_on_observations(
                        X=self.initial_model.transform_inputs(self.optimal_inputs),
                        Y=self.optimal_outputs,
                        noise=opt_noise
                    )
                else:
                    self.conditional_model = self.initial_model.condition_on_observations(
                        X=self.initial_model.transform_inputs(self.optimal_inputs),
                        Y=self.optimal_outputs,
                    )

        self.estimation_type = estimation_type
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qJointEntropySearch at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """
        if self.estimation_type == "LB":
            res = self._compute_lower_bound_information_gain(X)
        elif self.estimation_type == "MC":
            res = self._compute_monte_carlo_information_gain(X)
        else:
            raise ValueError(
                f"Estimation type {self.estimation_type} is not valid. "
                f"Please specify any of {ESTIMATION_TYPES}"
            )
        return res

    def _compute_lower_bound_information_gain(
        self, X: Tensor, return_parts: bool = False
    ) -> Tensor:
        r"""Evaluates the lower bound information gain at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """
        tkwargs = {
            "dtype": self.optimal_outputs.dtype,
            "device": self.optimal_outputs.device,
        }
        initial_posterior = self.initial_model.posterior(X, observation_noise=True)
        # need to check if there is a two-dimensional batch shape -
        # the sampled optima appear in the dimension right after
        batch_shape = X.shape[:-2]
        sample_dim = len(batch_shape)
        # We DISREGARD the additional constant term.
        initial_entropy = 0.5 * \
            torch.logdet(initial_posterior.mvn.lazy_covariance_matrix)

        # initial_entropy of shape batch_size or batch_size x num_models if FBGP
        # first need to unsqueeze the sample dim (first after batch dim) and then the two last
        initial_entropy = (
            initial_entropy.unsqueeze(sample_dim).unsqueeze(-1).unsqueeze(-1)
        )

        # Compute the mixture mean and variance
        posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=True
        )
        noiseless_var = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=False
        ).variance

        mean_m = posterior_m.mean
        if not self.maximize:
            mean_m = -mean_m
        variance_m = posterior_m.variance

        check_no_nans(variance_m)
        # get stdv of noiseless variance
        stdv = noiseless_var.sqrt()
        # batch_shape x 1
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )
        normalized_mvs = (self.optimal_outputs - mean_m) / stdv
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        ratio = pdf_mvs / cdf_mvs
        var_truncated = noiseless_var * (
            1 - (normalized_mvs + ratio) * ratio
        ).clamp_min(CLAMP_LB)

        var_truncated = var_truncated + (variance_m - noiseless_var)
        conditional_entropy = 0.5 * torch.log(var_truncated)

        # Shape batch_size x num_optima x [num_models if FB] x q x num_outputs
        # squeeze the num_outputs dim (since it's 1)
        entropy_reduction = (
            initial_entropy - conditional_entropy.sum(dim=-2, keepdim=True)
        ).squeeze(-1)
        # average over the number of optima and squeeze the q-batch

        entropy_reduction = entropy_reduction.mean(dim=sample_dim).squeeze(-1)
        return entropy_reduction

    def _compute_monte_carlo_variables(self, posterior):
        """Retrieved the monte carlo samples  and their log probabilities from the posterior.

        Args:
            posterior: The posterior distribution.

        Returns:
            A two-element tuple containing:
            - samples: a num_optima x batch_shape x num_mc_samples x q x 1 tensor of samples drawn from the posterior.
            - samples_log_prob: a num_optima x batch_shape x num_mc_samples x q x 1 tensor of associated probabilities.
        """
        samples = self.get_posterior_samples(posterior)
        samples_log_prob = (
            posterior.mvn.log_prob(samples.squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        )
        return samples, samples_log_prob

    def _compute_monte_carlo_information_gain(
        self, X: Tensor, return_parts: bool = False
    ) -> Tensor:
        r"""Evaluates the lower bound information gain at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
                `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """
        tkwargs = {
            "dtype": self.optimal_outputs.dtype,
            "device": self.optimal_outputs.device,
        }

        initial_posterior = self.initial_model.posterior(X, observation_noise=True)

        batch_shape = X.shape[:-2]
        sample_dim = len(batch_shape)
        # We DISREGARD the additional constant term.
        initial_entropy = MC_ADD_TERM + 0.5 * \
            torch.logdet(initial_posterior.mvn.lazy_covariance_matrix)

        # initial_entropy of shape batch_size or batch_size x num_models if FBGP
        # first need to unsqueeze the sample dim (first after batch dim) and then the two last
        initial_entropy = (
            initial_entropy.unsqueeze(sample_dim).unsqueeze(-1).unsqueeze(-1)
        )

        # Compute the mixture mean and variance
        posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=True
        )
        noiseless_var = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=False
        ).variance

        mean_m = posterior_m.mean
        if not self.maximize:
            mean_m = -mean_m
        variance_m = posterior_m.variance.clamp_min(CLAMP_LB)
        conditional_samples, conditional_logprobs = self._compute_monte_carlo_variables(
            posterior_m
        )

        normalized_samples = (conditional_samples - mean_m) / variance_m.sqrt()
        # Correlation between noisy observations and noiseless values f
        rho = (noiseless_var / variance_m).sqrt()

        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )
        # prepare max value quantities and re-scale as required
        normalized_mvs = (self.optimal_outputs - mean_m) / noiseless_var.sqrt()
        mvs_rescaled_mc = (normalized_mvs - rho * normalized_samples) / (1 - rho**2)
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        cdf_rescaled_mvs = normal.cdf(mvs_rescaled_mc).clamp_min(CLAMP_LB)
        mv_ratio = cdf_rescaled_mvs / cdf_mvs

        log_term = torch.log(mv_ratio) + conditional_logprobs
        conditional_entropy = -(mv_ratio * log_term).mean(0)
        entropy_reduction = (
            initial_entropy - conditional_entropy.sum(dim=-2, keepdim=True)
        ).squeeze(-1)

        # average over the number of optima and squeeze the q-batch
        entropy_reduction = entropy_reduction.mean(dim=sample_dim).squeeze(-1)

        return entropy_reduction


class qExploitJointEntropySearch(qJointEntropySearch):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.
    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal input-output pairs.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        exploit_fraction: float = 0.1,
        estimation_type: str = "LB",
        num_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        r"""gamma-exploit Joint entropy search acquisition function.
        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k. The number `K` is specified the variable
                `num_constraints`.
            pareto_sets: A `num_pareto_samples x num_pareto_points x d`-dim Tensor.
            pareto_fronts: A `num_pareto_samples x num_pareto_points x M`-dim Tensor.
            hypercell_bounds:  A `num_pareto_samples x 2 x J x (M + K)`-dim
                Tensor containing the hyper-rectangle bounds for integration. In the
                unconstrained case, this gives the partition of the dominated space.
                In the constrained case, this gives the partition of the feasible
                dominated space union the infeasible space.
            condition_noiseless: whether or not to condition on noiseless or noisy optimal
            observations. This is the main differentiator between 
                "Joint Entropy Search for Multi-Objective Bayesian Optimization" (False) 
                "Joint Entropy Search for Maximally-Informed Bayesian Optimization"(True)
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "Noiseless", "Lower bound" or "Monte Carlo".
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy i.e. the LB2
                strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples=64,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.
        """
        super().__init__(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            condition_noiseless=condition_noiseless,
            posterior_transform=posterior_transform,
            maximize=maximize,
            X_pending=X_pending,
            exploit_fraction=exploit_fraction,
            estimation_type=estimation_type,
            num_samples=num_samples,
            **kwargs
        )
        self.exploit = torch.rand([1]) < exploit_fraction

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute joint entropy search at the design points `X`, iterating between
        greedy and non-greedy depending on the iteration.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
                `d`-dim design points each.
        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """

        if self.exploit:

            # TODO ensure that this is no longer necessary
            if is_fully_bayesian(self.model):
                X = X.squeeze(1)
                res = self.initial_model.posterior(X).mean.squeeze(-1).squeeze(-1).T
            else:
                res = self.initial_model.posterior(X).mean.squeeze(-1).squeeze(-1)
        else:
            res = self._compute_lower_bound_information_gain(X)

        return res