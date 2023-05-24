#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Active learning acquisition functions.

.. [Seo2014activedata]
    S. Seo, M. Wallat, T. Graepel, and K. Obermayer. Gaussian process regression:
    Active data selection and test point rejection. IJCNN 2000.

.. [Chen2014seqexpdesign]
    X. Chen and Q. Zhou. Sequential experimental designs for stochastic kriging.
    Winter Simulation Conference 2014.

.. [Binois2017repexp]
    M. Binois, J. Huang, R. B. Gramacy, and M. Ludkovski. Replication or
    exploration? Sequential design for stochastic simulation experiments.
    ArXiv 2017.
"""

from __future__ import annotations

from typing import Optional

from math import pi
from copy import deepcopy

import torch
from torch.quasirandom import SobolEngine
from botorch import settings
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.models.model import Model

from botorch.models.utils import check_no_nans
from botorch.models.utils import fantasize as fantasize_flag
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from torch import Tensor
from gpytorch.mlls import ExactMarginalLogLikelihood

# FOR PLOTTING
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


MCMC_DIM = -3  # Location of the MCMC batch dimension
SAMPLES_DIM = -4  # Location of the optimal samples batch dim
CLAMP_LB = 1e-6


class qNegIntegratedPosteriorVariance(AnalyticAcquisitionFunction):
    r"""Batch Integrated Negative Posterior Variance for Active Learning.

    This acquisition function quantifies the (negative) integrated posterior variance
    (excluding observation noise, computed using MC integration) of the model.
    In that, it is a proxy for global model uncertainty, and thus purely focused on
    "exploration", rather the "exploitation" of many of the classic Bayesian
    Optimization acquisition functions.

    See [Seo2014activedata]_, [Chen2014seqexpdesign]_, and [Binois2017repexp]_.
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        sampler: Optional[MCSampler] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        r"""q-Integrated Negative Posterior Variance.

        Args:
            model: A fitted model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior variance. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration of the posterior variance.
            sampler: The sampler used for drawing fantasy samples. In the basic setting
                of a standard GP (default) this is a dummy, since the variance of the
                model after conditioning does not actually depend on the sampled values.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A `n' x d`-dim Tensor of `n'` design points that have
                points that have been submitted for function evaluation but
                have not yet been evaluated.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        if sampler is None:
            # If no sampler is provided, we use the following dummy sampler for the
            # fantasize() method in forward. IMPORTANT: This assumes that the posterior
            # variance does not depend on the samples y (only on x), which is true for
            # standard GP models, but not in general (e.g. for other likelihoods or
            # heteroskedastic GPs using a separate noise model fit on data).
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        self.sampler = sampler
        self.X_pending = X_pending
        self.register_buffer("mc_points", mc_points)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # Construct the fantasy model (we actually do not use the full model,
        # this is just a convenient way of computing fast posterior covariances
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=True
        )

        bdims = tuple(1 for _ in X.shape[:-2])
        if self.model.num_outputs > 1:
            # We use q=1 here b/c ScalarizedObjective currently does not fully exploit
            # LinearOperator operations and thus may be slow / overly memory-hungry.
            # TODO (T52818288): Properly use LinearOperators in scalarize_posterior
            mc_points = self.mc_points.view(-1, *bdims, 1, X.size(-1))
        else:
            # While we only need marginal variances, we can evaluate for q>1
            # b/c for GPyTorch models lazy evaluation can make this quite a bit
            # faster than evaluting in t-batch mode with q-batch size of 1
            mc_points = self.mc_points.view(*bdims, -1, X.size(-1))

        # evaluate the posterior at the grid points
        with settings.propagate_grads(True):
            posterior = fantasy_model.posterior(
                mc_points, posterior_transform=self.posterior_transform
            )

        neg_variance = posterior.variance.mul(-1.0)

        if self.posterior_transform is None:
            # if single-output, shape is 1 x batch_shape x num_grid_points x 1
            return neg_variance.mean(dim=-2).squeeze(-1).squeeze(0)
        else:
            # if multi-output + obj, shape is num_grid_points x batch_shape x 1 x 1
            return neg_variance.mean(dim=0).squeeze(-1).squeeze(-1)


class PairwiseMCPosteriorVariance(MCAcquisitionFunction):
    r"""Variance of difference for Active Learning

    Given a model and an objective, calculate the posterior sample variance
    of the objective on the difference of pairs of points. See more implementation
    details in `forward`. This acquisition function is typically used with a
    pairwise model (e.g., PairwiseGP) and a likelihood/link function
    on the pair difference (e.g., logistic or probit) for pure exploration
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Pairwise Monte Carlo Posterior Variance

        Args:
            model: A fitted model.
            objective: An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit.) applied on the difference of (usually 1-d)
                two samples. Can be implemented via GenericMCObjective.
            sampler: The sampler used for drawing MC samples.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=None
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PairwiseMCPosteriorVariance on the candidate set `X`.

        Args:
            X: A `batch_size x q x d`-dim Tensor. q should be a multiple of 2.

        Returns:
            Tensor of shape `batch_size x q` representing the posterior variance
            of link function at X that active learning hopes to maximize
        """
        if X.shape[-2] == 0 or X.shape[-2] % 2 != 0:
            raise RuntimeError(
                "q must be a multiple of 2 for PairwiseMCPosteriorVariance"
            )

        # The output is of shape batch_shape x 2 x d
        # For PairwiseGP, d = 1
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # num_samples x batch_shape x 2 x d

        # The output is of shape num_samples x batch_shape x q/2 x d
        # assuming the comparison is made between the 2 * i and 2 * i + 1 elements
        samples_diff = samples[..., ::2, :] - samples[..., 1::2, :]
        mc_var = self.objective(samples_diff).var(dim=0)
        mean_mc_var = mc_var.mean(dim=-1)

        return mean_mc_var


class BALM(AcquisitionFunction):
    def __init__(
            self,
            model: Model,
            **kwargs: Any
    ) -> None:
        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        return posterior.variance.mean(dim=MCMC_DIM).squeeze(-1)


class QBMGP(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        **kwargs: Any
    ) -> None:
        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        posterior_mean = posterior.mean
        marg_mean = posterior_mean.mean(dim=MCMC_DIM, keepdim=True)
        var_of_mean = torch.pow(
            marg_mean - posterior_mean, 2).mean(dim=MCMC_DIM, keepdim=True)
        mean_of_var = posterior.variance.mean(dim=MCMC_DIM, keepdim=True)
        return (var_of_mean + mean_of_var).squeeze(-1).squeeze(-1)


class BQBC(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        **kwargs: Any
    ) -> None:

        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        posterior_mean = posterior.mean
        marg_mean = posterior_mean.mean(dim=MCMC_DIM, keepdim=True)
        var_of_mean = torch.pow(marg_mean - posterior_mean, 2)
        return var_of_mean.squeeze(-1).squeeze(-1)


class StatisticalDistance(AcquisitionFunction):
    """Statistical distance-based Bayesian Active Learning

    Args:
        AcquisitionFunction ([type]): [description]
    """

    def __init__(
        self,
        model: Model,
        noisy_distance: bool = True,
        distance_metric: str = 'HR',
        estimate: str = 'MM',
        num_samples: int = 256,
        **kwargs: Any
    ) -> None:

        super().__init__(model)
        self.noisy_distance = noisy_distance
        available_dists = ['JS', 'KL', 'WS', 'BC', 'HR']
        if distance_metric not in available_dists:
            raise ValueError(f'Distance metric {distance_metric}'
                             f'Not available. Choose any of {available_dists}')
        self.distance = DISTANCE_METRICS[distance_metric]
        self.estimate = estimate
        self.distance_metric = distance_metric
        if estimate == 'MC':
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=1, scramble=True)
            samples = sobol.draw(num_samples)
            self.base_samples = torch.distributions.Normal(
                loc=0, scale=1).icdf(samples).squeeze(-1)

    def _compute(self, X: Tensor, mean_only: bool = False, var_only: bool = False) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=self.noisy_distance)

        cond_means = posterior.mean
        marg_mean = cond_means.mean(MCMC_DIM).unsqueeze(-1)
        cond_variances = posterior.variance
        marg_variance = cond_variances.mean(MCMC_DIM).unsqueeze(-1)

        dist_al = self.distance(cond_means, marg_mean, cond_variances, marg_variance)

        # MCMC dim mean is already taken
        return dist_al.mean(MCMC_DIM).squeeze(-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X.unsqueeze(
            MCMC_DIM), observation_noise=self.noisy_distance)
        cond_means = posterior.mean
        marg_mean = cond_means.mean(MCMC_DIM, keepdim=True)
        cond_variances = posterior.variance
        marg_variance = cond_variances.mean(MCMC_DIM, keepdim=True)
        if self.estimate == 'MM':
            dist_al = self.distance(cond_means, marg_mean,
                                    cond_variances, marg_variance)
        else:
            if self.distance_metric == 'HR':
                dist_al = _compute_hellinger_distance_mc(
                    cond_means, cond_variances, cond_means, cond_variances, self.base_samples)

            elif self.distance_metric == 'WS':
                dist_al = _compute_wasserstein_distance_mc(
                    cond_means, cond_variances, cond_means, cond_variances, self.base_samples)

        return dist_al.squeeze(-1).mean(-1).squeeze(1)


class JointSelfCorrecting(AcquisitionFunction):
    """Not really the correct name since it is not actually entropy search,
    but it's quite nice anyway.

    Args:
        AcquisitionFunction ([type]): [description]
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        noisy_distance: bool = True,
        condition_noiseless: bool = True,
        distance_metric: str = 'HR',
        estimate: str = 'MM',
        num_samples: int = 256,
        **kwargs: Any
    ) -> None:
        super().__init__(model)
        self.condition_noiseless = condition_noiseless
        self.initial_model = model
        self.posterior_transform = posterior_transform
        self.noisy_distance = noisy_distance
        self.num_models = optimal_inputs.shape[1]
        self.estimate = estimate
        available_dists = ['WS','HR']
        if distance_metric not in available_dists:
            raise ValueError(f'Distance metric {distance_metric}'
                             f'Not available. Choose any of {available_dists}')
        self.distance = DISTANCE_METRICS[distance_metric]
        if estimate == 'MC':
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=1, scramble=True)
            samples = sobol.draw(num_samples)
            self.base_samples = torch.distributions.Normal(
                loc=0, scale=1).icdf(samples).squeeze(-1)
            # self.base_samples = torch.distributions.Normal(loc=0, scale=1).rsample(
            #   sample_shape=torch.Size([num_samples]))
        self.optimal_inputs = optimal_inputs.unsqueeze(-2)
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)
        self.posterior_transform = posterior_transform
        self.initial_model = model
        self.distance_metric = distance_metric
    
        with fantasize_flag():
            with settings.propagate_grads(False):
                post_ps = self.initial_model.posterior(
                    self.model.train_inputs[0], observation_noise=False
                )
                sample_idx = 0

            # Warning occurs due to the non-conventional batch shape (I think?)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # conditional (batch) model of shape (num_models) x num_optima_per_model
                self.conditional_model = self.initial_model.condition_on_observations(
                    X=self.initial_model.transform_inputs(self.optimal_inputs),
                    Y=self.optimal_outputs,
                    noise=CLAMP_LB
                    * torch.ones_like(self.optimal_outputs)
                )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=self.noisy_distance)

        noiseless_var = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=False
        ).variance

        mean_m = posterior_m.mean
        variance_m = posterior_m.variance.clamp_min(CLAMP_LB)

        check_no_nans(variance_m)
        # get stdv of noiseless variance
        stdv = noiseless_var.sqrt()
        # batch_shape x 1
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )
        # prepare max value quantities required by ScoreBO
        normalized_mvs = (self.optimal_outputs - mean_m) / stdv
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        mean_truncated = mean_m - stdv * pdf_mvs / cdf_mvs
        # This is the noiseless variance (i.e. the part that gets truncated)
        var_truncated = noiseless_var * \
            (1 - normalized_mvs * pdf_mvs / cdf_mvs - torch.pow(pdf_mvs / cdf_mvs, 2))
        var_truncated = var_truncated + (variance_m - noiseless_var)

        prev_m = self.initial_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=self.noisy_distance)
        reference_mean = prev_m.mean
        reference_var = prev_m.variance
        
        if self.estimate == 'MM':
            dist = _compute_disagreement_mm(
                mean_truncated, var_truncated, reference_mean, reference_var, self.distance)

        elif self.estimate == 'MC':
            if self.distance_metric == 'HR':
                dist = _compute_hellinger_distance_mc(
                    mean_truncated, var_truncated, reference_mean, reference_var, self.base_samples)

            elif self.distance_metric == 'WS':
                dist = _compute_wasserstein_distance_mc(
                    mean_truncated, var_truncated, reference_mean, reference_var, self.base_samples)

        return dist.mean(SAMPLES_DIM).squeeze(-1).squeeze(-1)


def _compute_disagreement_mm_old(mean_truncated, var_truncated, mm_distance):
    marg_truncated_mean = mean_truncated.mean(
        MCMC_DIM, keepdim=True).mean(SAMPLES_DIM, keepdim=True)

    num_mcmc_samples = mean_truncated.shape[SAMPLES_DIM] * \
        mean_truncated.shape[MCMC_DIM]
    t1 = var_truncated.sum(dim=[SAMPLES_DIM, MCMC_DIM], keepdim=True) / num_mcmc_samples
    t2 = mean_truncated.pow(2).sum(
        dim=[SAMPLES_DIM, MCMC_DIM], keepdim=True) / num_mcmc_samples
    t3 = -(mean_truncated.sum(dim=[SAMPLES_DIM, MCMC_DIM],
                              keepdim=True) / num_mcmc_samples).pow(2)
    marg_truncated_var = t1 + t2 + t3

    # marg_truncated_var = var_truncated.mean(
    #    MCMC_DIM, keepdim=True).mean(SAMPLES_DIM, keepdim=True)
    dist = mm_distance(mean_truncated, marg_truncated_mean,
                       var_truncated, marg_truncated_var)
    return dist


def _compute_disagreement_mm(mean_cond, var_cond, reference_mean, reference_var, mm_distance):
    marg_reference_mean = reference_mean.mean([SAMPLES_DIM, MCMC_DIM], keepdim=True)
    t1 = reference_var.mean(dim=[SAMPLES_DIM, MCMC_DIM], keepdim=True)
    t2 = reference_mean.pow(2).mean(dim=[SAMPLES_DIM, MCMC_DIM], keepdim=True)
    t3 = -(reference_mean.mean(dim=[SAMPLES_DIM, MCMC_DIM], keepdim=True)).pow(2)
    marg_reference_var = t1 + t2 + t3

    dist = mm_distance(mean_cond, marg_reference_mean,
                       var_cond, marg_reference_var)
    return dist


def _compute_hellinger_distance_mc(conditional_mean, conditional_var, reference_mean, reference_var, base_normal_samples):
    mc_est = torch.zeros_like(conditional_mean)
    # Draw samples from the conditional (i.e. optimum-conditioned) distribution
    # and evaluate on the ratio sqrt(pdf_marginal / pdf_conditional)
    conditional_samples = base_normal_samples * \
        torch.sqrt(conditional_var) + conditional_mean

    all_conditional_dists = torch.distributions.Normal(
        conditional_mean.unsqueeze(-1).unsqueeze(-1), torch.sqrt(conditional_var).unsqueeze(-1).unsqueeze(-1))
    all_marginal_dists = torch.distributions.Normal(
        reference_mean.unsqueeze(-1).unsqueeze(-1), torch.sqrt(reference_var).unsqueeze(-1).unsqueeze(-1))

    conditional_samples = conditional_samples.unsqueeze(-1).unsqueeze(-1)
    # mixture_samples = conditional_mean_samples.transpose(MCMC_DIM - 2, -1)
    # mixture_samples = mixture_samples.transpose(SAMPLES_DIM - 2, -2)
    # mixture_probs = torch.exp(all_conditional_dists.log_prob(mixture_samples))

    num_models = conditional_mean.shape[MCMC_DIM]
    num_optima = conditional_mean.shape[SAMPLES_DIM]

    cond_probs = torch.exp(all_conditional_dists.log_prob(conditional_samples))

    for model in range(num_models):
        for opt in range(num_optima):
            # the probability of the mode
            marg_probs = torch.exp(all_marginal_dists.log_prob(
                conditional_samples[:, opt, model, : , :].unsqueeze(1).unsqueeze(2))).mean(dim=[1, 2])
            # estimating the distance from ONE conditional to the marginal
            mc_est[:, opt, model, : , :] = torch.sqrt(
                marg_probs / cond_probs[:, opt, model, : , :]).squeeze(-1).squeeze(-1).mean(-1, keepdim=True).clamp_max(1)

    return torch.sqrt(1 - mc_est)


def _compute_wasserstein_distance_mc(conditional_mean, conditional_var, reference_mean, reference_var, base_normal_samples, order=2):
    base_unif_samples = torch.distributions.Normal(0, 1).cdf(base_normal_samples)
    # Draw samples from the conditional (i.e. optimum-conditioned) distribution
    # and evaluate on the ratio sqrt(pdf_marginal / pdf_conditional)

    all_conditional_dists = torch.distributions.Normal(
        conditional_mean, torch.sqrt(conditional_var))
    all_marginal_dists = torch.distributions.Normal(
        reference_mean, torch.sqrt(reference_var))

    cond_probs = all_conditional_dists.icdf(base_unif_samples)
    marg_probs = all_marginal_dists.icdf(base_unif_samples).mean([-4, -3], keepdim=True)
    integral_term = torch.pow(torch.abs(cond_probs - marg_probs), order)
    dist = torch.pow(integral_term.mean(-1, keepdim=True), 1 / order)
    return dist
    # return mc_dist


def _compute_hellinger_distance_mc_mix(mean_truncated, var_truncated, base_normal_samples):
    # all the samples from the mixture - for each model, num_samples have been drawn
    # all have the sample probability of being drawn, since they are base samples?
    conditional_mean_samples = base_normal_samples * \
        torch.sqrt(var_truncated) + mean_truncated

    all_conditional_dists = torch.distributions.Normal(
        mean_truncated.unsqueeze(-1).unsqueeze(-1), torch.sqrt(var_truncated).unsqueeze(-1).unsqueeze(-1))

    conditional_mean_samples = conditional_mean_samples.unsqueeze(-1).unsqueeze(-1)
    mixture_samples = conditional_mean_samples.transpose(MCMC_DIM - 2, -1)
    mixture_samples = mixture_samples.transpose(SAMPLES_DIM - 2, -2)
    mixture_probs = torch.exp(all_conditional_dists.log_prob(mixture_samples))

    # Option 1
    mixture_probs = mixture_probs.mean(
        dim=[MCMC_DIM - 2, SAMPLES_DIM - 2], keepdim=True)
    mixture_probs = mixture_probs.transpose(
        MCMC_DIM - 2, -1).transpose(SAMPLES_DIM - 2, -2)

    # Option 2
    mixture_probs = mixture_probs.mean(dim=[-2, -1], keepdim=True)
    #raise SystemExit
    sample_probs = torch.exp(all_conditional_dists.log_prob(conditional_mean_samples))
    # target: Batch_dim x num_optima x num_models x 1 x num_MC x num_optima x num_models
    mc_dist = torch.sqrt(mixture_probs / sample_probs).squeeze(-1).squeeze(-1)
    mc_dist = (1 - torch.mean(mc_dist, dim=-1, keepdim=True)).clamp_min(0).sqrt()
    return mc_dist


def wasserstein_distance_mm(mean_x, mean_y, var_x, var_y):
    mean_term = torch.pow(mean_x - mean_y, 2)
    # rounding errors sometimes occur, where the var term is ~-1e-16
    var_term = (var_x + var_y - 2 * torch.sqrt(var_x * var_y)).clamp_min(0)
    return torch.sqrt(mean_term + var_term)


def wasserstein_distance(mean_x, mean_y, var_x, var_y):
    # Duplicate the samples so that we can compare all the dists against each other
    mean_y = mean_y.transpose(-4, -2).transpose(-3, -1)
    var_y = var_y.transpose(-4, -2).transpose(-3, -1)
    mean_term = torch.pow(mean_x - mean_y, 2)
    # rounding errors sometimes occur, where the var term is ~-1e-16
    var_term = (var_x + var_y - 2 * torch.sqrt(var_x * var_y)).clamp_min(0)

    return torch.sqrt((mean_term + var_term).mean(dim=[-2, -1], keepdim=True))

def hellinger_distance(mean_x, mean_y, var_x, var_y):
    exp_term = -0.25 * torch.pow(mean_x - mean_y, 2) / (var_x + var_y)
    mult_term = torch.sqrt(2 * torch.sqrt(var_x * var_y) / (var_x + var_y))
    return torch.sqrt(1 - mult_term * torch.exp(exp_term))



DISTANCE_METRICS = {
    'WS': wasserstein_distance,
    'HR': hellinger_distance,
}
