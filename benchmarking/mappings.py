from functools import partial
from copy import copy

from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP, SingleTaskALGP
from botorch.test_functions import (
    Branin,
    Hartmann,
    Ackley,
    Rosenbrock,
)

from ax.modelbridge.registry import Models
from benchmarking.gp_sample.gp_task import GPTestFunction
from benchmarking.synthetic import (
    Gramacy1,
    Gramacy2,
    Higdon,
    Ishigami,
    Embedded,
    AdditiveEmbedded
)
from benchmarking.lassobench_task import LassoRealFunction
from benchmarking.pd1_task import PD1Function
from benchmarking.cosmo_task import CosmoFunction
from botorch.acquisition import (
    qMaxValueEntropy,
    qKnowledgeGradient,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.predictive_entropy_search import (
    qPredictiveEntropySearch
)
from botorch.acquisition.joint_entropy_search import (
    qJointEntropySearch,
    qExploitJointEntropySearch,
)

from botorch.acquisition.active_learning import (
    JointSelfCorrecting,
    StatisticalDistance,
    QBMGP,
    BALM,
    BQBC,
)

from ax.models.torch.fully_bayesian import (
    single_task_pyro_model,
)
from botorch.models.fully_bayesian import (
    WarpingPyroModel,
    SaasPyroModel,
    SingleTaskMeanPyroModel,
    ActiveLearningPyroModel,
    ActiveLearningWithoutMeanPyroModel,
    ActiveLearningWithOutputscalePyroModel,
    FixedAdditiveSamplingModel,
    SaasFullyBayesianSingleTaskGP,
    WarpingFullyBayesianSingleTaskGP,
)
from ax.models.torch.fully_bayesian_model_utils import (
    _get_active_learning_gpytorch_model,
    _get_single_task_gpytorch_model,
)
import torch


# Gets the desired test function without initiating all of them (which requires a bit of trickery)
def get_test_function(name: str, noise_std: float = 0, seed: int = 0):

    TEST_FUNCTIONS = {
        'gramacy1': Gramacy1,
        'higdon': Higdon,
        'gramacy2': Gramacy2,
        'ishigami': Ishigami,
        'active_branin': Branin,
        'active_hartmann6': Hartmann,
        'branin': Branin,
        'ackley4': Ackley,
        'ackley4_25': Embedded,
        'rosenbrock2': Rosenbrock,
        'rosenbrock4': Rosenbrock,
        'hartmann3': Hartmann,
        'hartmann4': Hartmann,
        'hartmann6': Hartmann,
        'hartmann6_25': Embedded,
        'lasso_dna': LassoRealFunction,
        'cosmo': CosmoFunction,
        'gp_8dim': GPTestFunction,
        'gp_2dim': GPTestFunction,
        'gp_2_2_2dim': AdditiveEmbedded,
        'gp_2_2_2_2_2dim': AdditiveEmbedded,
        'pd1_wmt': PD1Function,
        'pd1_lm1b': PD1Function,
        'pd1_cifar': PD1Function,
        
    }

    test_function = TEST_FUNCTIONS[name]
    default_args = {'noise_std': noise_std, 'negate': True}

    extra_args = {
        'ackley4_25': {'dim': 25, 'embedded_args': {'name': 'ackley4'}},
        'hartmann6_25': {'dim': 25, 'embedded_args': {'name': 'hartmann6'}},
        'lasso_dna': dict(seed=seed, pick_data='dna'),
        'gp_2dim': dict(seed=seed, dim=2),
        'gp_8dim': dict(seed=seed, dim=8),
        'cosmo': dict(seed=seed),
        'gp_2_2_2dim': {'dim': 6, 'embedded_args': {'name': ['gp_2dim'] * 3, 'seed': list(range(3))}},
        'gp_2_2_2_2_2dim': {'dim': 10, 'embedded_args': {'name': ['gp_2dim'] * 5, 'seed': list(range(5))}},
        'hartmann3': dict(dim=3),
        'rosenbrock2': dict(dim=2),
        'rosenbrock4': dict(dim=4),
        'hartmann4': dict(dim=4),
        'hartmann6': dict(dim=6),
        'ackley4': dict(dim=4),
        'pd1_wmt': dict(negate=False, seed=seed, task_name='translatewmt_xformer_64'),
        'pd1_lm1b': dict(negate=False, seed=seed, task_name='lm1b_transformer_2048'),
        'pd1_cifar': dict(negate=False, seed=seed, task_name='cifar100_wideresnet_2048'),
    }

    default_args.update(extra_args.get(name, {}))

    def construct_test_function(func, func_args):
        if 'embedded_args' in func_args:
            other_args = copy(func_args['embedded_args'])
            other_args.pop('name')

            if isinstance(func_args['embedded_args']['name'], list):
                functions = []
                names = func_args['embedded_args']['name']
                for idx, name in enumerate(names):
                    # pick out the i:th element of the remaining args
                    args_idx = {key: value[idx] for key, value in other_args.items()}
                    functions.append(get_test_function(name, **args_idx))
                func_args['functions'] = functions

            else:
                function = get_test_function(
                    func_args['embedded_args']['name'], **other_args)
                func_args['function'] = function
            func_args.pop('embedded_args')

        return func(**func_args)

    return construct_test_function(test_function, default_args)


ACQUISITION_FUNCTIONS = {
    'NEI': qNoisyExpectedImprovement,
    'JES-e': qExploitJointEntropySearch,
    'GIBBON': qLowerBoundMaxValueEntropy,
    'ScoreBO_J': JointSelfCorrecting,
    'PES': qPredictiveEntropySearch,
    'SAL': StatisticalDistance,
    'BALM': BALM,
    'QBMGP': QBMGP,
    'BALM': BALM,
    'BQBC': BQBC,
    
    

}

MODELS = {
    'FixedNoiseGP': FixedNoiseGP,
    'SingleTaskGP': SingleTaskGP,
}


GP_PRIORS = {
    'SCoreBO_nuts_mean': dict(
        pyro_model=SingleTaskMeanPyroModel()),
    'SCoreBO_al_bo': dict(
        pyro_model=ActiveLearningWithOutputscalePyroModel()),
    'SCoreBO_al_op': dict(
        pyro_model=ActiveLearningWithoutMeanPyroModel()),
    'ScoreBO_saas': dict(
        pyro_model=SaasPyroModel()),
    'SCoreBO_al': dict(
        pyro_model=ActiveLearningPyroModel()),
    'ScoreBO_add': dict(
        pyro_model=FixedAdditiveSamplingModel()),
    'ScoreBO_warp': dict(
        pyro_model=WarpingPyroModel()),
}

MODEL_CLASSES = {
    'SCoreBO_nuts_mean': SaasFullyBayesianSingleTaskGP,
    'SCoreBO_al': SaasFullyBayesianSingleTaskGP,
    'SCoreBO_al_bo': SaasFullyBayesianSingleTaskGP,
    'ScoreBO_add': SaasFullyBayesianSingleTaskGP,
    'ScoreBO_saas': SaasFullyBayesianSingleTaskGP,
    'SCoreBO_al_op': SaasFullyBayesianSingleTaskGP,
    'ScoreBO_warp': WarpingFullyBayesianSingleTaskGP,

}


def get_warm_start_args(model, model_name):
    if model_name == 'ScoreBO_add':
        warm_start_state = model.get_warm_start_state()
        return {'warm_start_state': warm_start_state}
    else:
        return {}
