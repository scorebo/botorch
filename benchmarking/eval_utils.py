import torch
import numpy as np
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean, PolynomialMean
from botorch.models.transforms import Warp
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import AdditiveKernel
from botorch.utils.transforms import unnormalize


def get_model_hyperparameters(model, current_data, scale_hyperparameters=True, objective=None, acquisition=None):

    has_outputscale = isinstance(model.covar_module, ScaleKernel)
    has_mean = isinstance(model.mean_module, ConstantMean)
    is_poly = isinstance(model.mean_module, PolynomialMean)
    has_warping = hasattr(model, "input_transform") and isinstance(
        model.input_transform, Warp)

    def tolist(l): return l.detach().to(torch.float32).numpy().tolist()
    def d(l): return l.detach().to(torch.float32).numpy().flatten()
    hp_dict = {}

    data_mean = current_data.mean()

    training_data = model.train_inputs[0]

    from torch.quasirandom import SobolEngine
    draws = SobolEngine(dimension=training_data.shape[-1]).draw(512)
    
    if scale_hyperparameters:
        data_variance = current_data.var()
    
    else:
        data_variance = torch.Tensor([1])

    if has_warping:
        hp_dict['concentration0'] = tolist(model.input_transform.concentration0)
        hp_dict['concentration1'] = tolist(model.input_transform.concentration1)

    if has_outputscale:
        hp_dict['outputscale'] = tolist(model.covar_module.outputscale * data_variance)
        if isinstance(model.covar_module.base_kernel, AdditiveKernel):
            hp_dict['lengthscales'] = tolist(
                model.covar_module.base_kernel.kernels[0].lengthscale)
            hp_dict['active_groups'] = []
            for kernel in model.covar_module.base_kernel.kernels:
                hp_dict['active_groups'].append(tolist(kernel.active_dims))
        else:
            hp_dict['lengthscales'] = tolist(model.covar_module.base_kernel.lengthscale)
        hp_dict['noise'] = tolist(model.likelihood.noise * data_variance)
    else:
        if isinstance(model.covar_module, AdditiveKernel):
            hp_dict['active_groups'] = []
            hp_dict['lengthscales'] = tolist(model.covar_module.kernels[0].lengthscale)
            hp_dict['noise'] = tolist(model.likelihood.noise)
            for kernel in model.covar_module.kernels:
                hp_dict['active_groups'].append(tolist(kernel.active_dims))
        else:
            hp_dict['lengthscales'] = tolist(model.covar_module.lengthscale)
            hp_dict['noise'] = tolist(model.likelihood.noise)

    if has_mean:
        hp_dict['mean'] = tolist(model.mean_module.constant
                                 * data_variance ** 0.5 - data_mean)
    if is_poly:
        hp_dict['const'] = tolist(model.mean_module.bias)
                                  #* data_variance ** 0.5 - data_mean)
        hp_dict['poly'] = tolist(model.mean_module.weights)
                                 #* data_variance ** 0.5)
    return hp_dict


def compute_rmse_and_nmll(ax_client, test_samples, objective, objective_name):
    TS_SPLIT = 10
    split_len = int(len(test_samples) / TS_SPLIT)
    loglik = 0
    rmse = 0
    for split_idx in range(TS_SPLIT):
        split_idx_low, split_idx_high = split_idx * \
            split_len, (1 + split_idx) * split_len
        test_sample_batch = test_samples[split_idx_low:split_idx_high]
        output = - \
            objective.evaluate_true(unnormalize(test_sample_batch, objective.bounds))
        y_transform = ax_client._generation_strategy.model.transforms['StandardizeY']
        y_mean, y_std = y_transform.Ymean[objective_name], y_transform.Ystd[objective_name]

        mu, cov = ax_client._generation_strategy.model.model.predict(test_sample_batch)
        mu_true = (mu * y_std + y_mean).flatten()

        model = ax_client._generation_strategy.model.model.surrogate.model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.eval()
        preds = model(test_sample_batch)
        norm_yvalid = (output - y_mean) / y_std

        norm_yvalid = norm_yvalid.flatten()
        #marg_dist = MultivariateNormal(predmean, predcov)
        #joint_loglik = -mll(marg_dist, norm_yvalid).mean()
        loglik = loglik - mll(preds, norm_yvalid).mean().item()
        rmse = rmse + torch.pow(output - mu_true, 2).mean().item()

    return loglik / TS_SPLIT, torch.Tensor([0]), rmse / TS_SPLIT

