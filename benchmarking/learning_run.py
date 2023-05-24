import os
from os.path import dirname, abspath, join
import sys
from time import time
from functools import partial
import json
from copy import copy

import hydra
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
import pandas as pd
import numpy as np

from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.ax_client import AxClient
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from omegaconf import DictConfig
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)

from copy import deepcopy
from ax.core.observation import ObservationFeatures
from benchmarking.synthetic import AdditiveEmbedded
from benchmarking.cosmo_task import CosmoFunction
from botorch.models import SingleTaskGP
from ax.models.torch.botorch_defaults import _get_acquisition_func

from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.prediction_utils import predict_at_point
from botorch.utils.transforms import unnormalize
from benchmarking.mappings import (
    get_test_function,
    get_warm_start_args,
    ACQUISITION_FUNCTIONS,
    MODELS,
    GP_PRIORS,
    MODEL_CLASSES
)
from benchmarking.eval_utils import (
    get_model_hyperparameters,
    compute_rmse_and_nmll
)
sys.path.append('.')

N_VALID_SAMPLES = int(2e3)
MIN_INFERRED_NOISE_LEVEL = 1e-4


@hydra.main(config_path='./../configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    print(cfg)
    torch.manual_seed(int(cfg.seed))
    acq_func = ACQUISITION_FUNCTIONS[cfg.algorithm.acq_func]

    prior_name = cfg.model.prior_name
    model_kwargs = GP_PRIORS[prior_name]
    if hasattr(cfg.model, 'gp') and (cfg.model.gp == 'Modular' or cfg.model.gp == 'trans_y'):
        model_kwargs.update(
            {
                "num_samples": cfg.model.num_samples,
                "warmup": cfg.model.warmup,
                "thinning": cfg.model.thinning,
                "disable_progbar": cfg.model.get("disable_progbar", False)
            }
        )

    model_class = MODEL_CLASSES.get(prior_name)
    if cfg.model.model_kwargs is not None:
        model_kwargs.update(dict(cfg.model.model_kwargs))

    refit_on_update = not hasattr(cfg.model, 'model_parameters')
    refit_params = {}

    if not refit_on_update:
        model_enum = Models.BOTORCH_MODULAR_NOTRANS
        if hasattr(cfg.benchmark, 'model_parameters'):
            params = cfg.benchmark.model_parameters
            refit_params['outputscale'] = Tensor(params.outputscale)
            refit_params['lengthscale'] = Tensor(params.lengthscale)

        # if the model also has fixed parameters, we will override with those
        if hasattr(cfg.model, 'model_parameters'):
            params = cfg.benchmark.model_parameters
            refit_params['outputscale'] = Tensor(params.outputscale)
            refit_params['lengthscale'] = Tensor(params.lengthscale)

    else:
        if cfg.model.get('notrans', False):
            model_enum = Models.BOTORCH_MODULAR_NOTRANS
        elif cfg.model.gp == 'trans_y':
            model_enum = Models.BOTORCH_MODULAR_TRANS_Y
        else:
            model_enum = Models.BOTORCH_MODULAR

    benchmark = cfg.benchmark.name

    test_function = get_test_function(
        benchmark, float(cfg.benchmark.noise_std), cfg.seed)

    if isinstance(test_function, (AdditiveEmbedded, CosmoFunction)):
        model_kwargs['num_groups'] = len(test_function.embedded_functions)
        model_kwargs['hyperparameters'] = {
            'noise' : Tensor([cfg.benchmark.noise_std ** 2]),
            'lengthscale' : Tensor(cfg.benchmark.model_parameters.lengthscale),
            'outputscale' : Tensor(cfg.benchmark.model_parameters.outputscale),
        }
        if cfg.model.get('oracle', False):
            model_kwargs['hyperparameters']['active_groups'] = Tensor(
                cfg.benchmark.model_parameters.active_groups)

    num_init = cfg.benchmark.num_init
    num_bo = cfg.benchmark.num_iters - num_init
    bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0)
    opt_setup = cfg.acq_opt

    refit_params = {}
    if hasattr(cfg.algorithm, 'acq_kwargs'):
        acq_func_kwargs = dict(cfg.algorithm.acq_kwargs)
    else:
        acq_func_kwargs = {}

    init_kwargs = {"seed": int(cfg.seed)}
    steps = [
        GenerationStep(
            model=Models.SOBOL,
            num_trials=num_init,
            # Otherwise, it's probably just SOBOL
            model_kwargs=init_kwargs,
        )]

    bo_step = GenerationStep(
        # model=model_enum,
        model=model_enum,
        # No limit on how many generator runs will be produced
        num_trials=num_bo,
        model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
            "surrogate": Surrogate(
                        botorch_model_class=model_class,
                        model_options=model_kwargs
            ),
            "botorch_acqf_class": acq_func,
            "acquisition_options": {**acq_func_kwargs},
            "refit_on_update": refit_on_update,
            "refit_params": refit_params
        },
        model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
            "optimizer_kwargs": dict(opt_setup)},
        },
    )
    steps.append(bo_step)

    def evaluate(parameters, seed=None):
        x = torch.tensor(
            [[parameters.get(f"x_{i+1}") for i in range(test_function.dim)]])
        if seed is not None:
            bc_eval = test_function.evaluate_true(x, seed=seed).squeeze().tolist()
        else:
            bc_eval = test_function(x).squeeze().tolist()
        # In our case, standard error is 0, since we are computing a synthetic function.
        return {benchmark: bc_eval}

    gs = GenerationStrategy(
        steps=steps
    )

    # Initialize the client - AxClient offers a convenient API to control the experiment
    ax_client = AxClient(generation_strategy=gs, verbose_logging=True)
    # Setup the experiment
    ax_client.create_experiment(
        name=cfg.experiment_name,
        parameters=[
            {
                "name": f"x_{i+1}",
                "type": "range",
                # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
                # Otherwise, the parameter would
                "bounds": bounds[:, i].tolist(),
                "value_type": 'float'
            }
            for i in range(test_function.dim)
        ],
        objectives={
            benchmark: ObjectiveProperties(minimize=False),
        },
    )

    loglik_arr = [0] * num_init
    joint_loglik_arr = [0] * num_init
    rmse_arr = [0] * num_init
    true_vals = [0.0] * (num_init)
    guess_vals = [0.0] * (num_init)

    best_guesses = []
    hyperparameters = {}
    likelihoods = {}
    likelihoods['other_likelihoods'] = []
    scale_hyperparameters = model_enum != Models.BOTORCH_MODULAR_NOTRANS
    bo_times = []

    for i in range(num_init + num_bo):
        if i >= num_init:
            start_time = time()
        parameters, trial_index = ax_client.get_next_trial()
        if i >= num_init:
            end_time = time()
            bo_times.append(end_time - start_time)
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=evaluate(parameters))
        # Unique sobol draws per seed and iteration, but identical for different acqs

        sobol = SobolEngine(dimension=test_function.dim,
                            scramble=True, seed=i + cfg.seed * 100)
        # crucially, these are already in [0, 1]
        test_samples = sobol.draw(n=N_VALID_SAMPLES)

        # These are for computing the active learning metrics
        current_data = ax_client.get_trials_data_frame()[benchmark].to_numpy()
        if (i >= num_init) and cfg.get('AL', False):
            print('Computing AL stuff')
            loglik, joint_loglik, rmse = compute_rmse_and_nmll(
                ax_client, test_samples, test_function, benchmark)
            loglik_arr.append(loglik)
            joint_loglik_arr.append(joint_loglik.item())
            rmse_arr.append(rmse)
            model = ax_client._generation_strategy.model.model.surrogate.model
            hps = get_model_hyperparameters(
                model, current_data, scale_hyperparameters=scale_hyperparameters, objective=test_function)
            hyperparameters[f'iter_{i}'] = hps

        elif (i >= num_init):
            model = ax_client._generation_strategy.model.model.surrogate.model

            acq = ax_client._generation_strategy.model.model.evaluate_acquisition_function
            hps = get_model_hyperparameters(
                model, current_data, scale_hyperparameters=scale_hyperparameters, objective=test_function, acquisition=acq)
            hyperparameters[f'iter_{i}'] = hps

        if cfg.algorithm.name != 'Dummy' and 'hpo' not in cfg.benchmark.name:
            guess_parameters, guess_trial_index = ax_client.get_best_guess()
            best_guesses.append(guess_parameters)

        # Update warm start arguments for next iteration if there are any
        if cfg.model.get('warm_start', False) and (i >= num_init):
            warm_start_options = get_warm_start_args(model, prior_name)
            ax_client._generation_strategy.model.model.surrogate.model_options.update(
                warm_start_options)

        if i >= num_init:

            results_df = ax_client.get_trials_data_frame()
            configs = torch.tensor(
                results_df.loc[:, ['x_' in col for col in results_df.columns]].to_numpy())

            if 'hpo' not in cfg.benchmark.name:
                true_vals.append(test_function.evaluate_true(
                    configs[i].unsqueeze(0)).item())
                results_df['True Eval'] = true_vals
                infer_values = None

            else:
                guess_parameters, guess_trial_index = ax_client.get_best_guess()
                guesses_tensor = Tensor([list(guess.values())
                                         for guess in best_guesses])
            if 'hpo' in cfg.benchmark.name:
                rounds = int(cfg.num_infer_rounds)
                num_data = len(results_df)
                infer = np.zeros(num_data)
                for i in range(cfg.num_infer_rounds):
                    res = evaluate(guess_parameters, seed=i)[cfg.benchmark.name]
                    print(res)
                    infer[i] = res
                results_df['Infer'] = np.flip(infer)

            if cfg.get('AL', False):
                results_df['MLL'] = loglik_arr
                results_df['JLL'] = joint_loglik_arr
                results_df['RMSE'] = rmse_arr
            elif (cfg.algorithm.name == 'Dummy') or 'hpo' in cfg.benchmark.name:
                pass
            else:
                guesses_tensor = Tensor([list(guess.values())
                                         for guess in best_guesses])
                guess_vals.append(test_function.evaluate_true(
                    guesses_tensor[i].unsqueeze(0)).item())
                results_df['Guess values'] = guess_vals

            os.makedirs(cfg.result_path, exist_ok=True)
            with open(f"{cfg.result_path}/{ax_client.experiment.name}_hps.json", "w") as f:
                json.dump(hyperparameters, f, indent=2)
            results_df.to_csv(f"{cfg.result_path}/{ax_client.experiment.name}.csv")
            try:
                pd.DataFrame(bo_times, columns=['runtime']).to_csv(
                    f"{cfg.result_path}/{ax_client.experiment.name}_times.csv", index=False)
            except:
                print('Could not save runtimes')
            try:
                pd.DataFrame(likelihoods).to_csv(
                    f"{cfg.result_path}/{ax_client.experiment.name}_likelihoods.csv", index=False)
            except:
                print('Could not save lieklhoodos')


if __name__ == '__main__':
    main()
