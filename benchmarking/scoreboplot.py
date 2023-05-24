import sys
import os
from os.path import join, dirname, isdir, abspath
from glob import glob
import json

from copy import copy
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.stats import sem
import pandas as pd
from pandas.errors import EmptyDataError


plt.rcParams['font.family'] = 'serif'

# Some constants for better estetics
# '#377eb8', '#ff7f00', '#4daf4a'
# '#f781bf', '#a65628', '#984ea3'
# '#999999', '#e41a1c', '#dede00'

BENCHMARK_PACKS = {
    'synthetic_bo': {'names': ('branin', 'hartmann3', 'hartmann4', 'hartmann6', 'rosenbrock2', 'rosenbrock4')},
    'synthetic_al': {'names': ('higdon', 'gramacy1', 'gramacy2', 'active_branin', 'ishigami', 'active_hartmann6')},
    'synthetic_al_rmse': {'names': ('higdon', 'gramacy1', 'gramacy2', 'active_branin', 'ishigami', 'active_hartmann6')},
    'gp_prior': {'names': (['gp_8dim'])},
    'saasbo': {'names': ('ackley4_25', 'hartmann6_25', 'lasso_dna'), 'best': (0, -3.3223, None)},
    'addgp_bo': {'names': ('gp_2_2_2dim', 'gp_2_2_2_2_2dim', 'cosmo')},
    'gtbo_syn': {'names': ('branin2', 'hartmann6', 'levy4')},
    'gtbo_real': {'names': ('lasso-dna', 'mopta08')},
}


COLORS = {

    'NEI_correct': 'k',
    'NEI_correct_name': 'k',
    'NEI_temp': '#984ea3',
    'NEI_wide': '#984ea3',
    'NEI_botorch': '#ff7f00',

    'ScoreBO_J_HR_wide': '#e41a1c',
    'ScoreBO_J_HR_botorch' : '#dede00',
    'ScoreBO_J': 'deeppink',
    'ScoreBO_J_HR_notrunc': 'red',
    'Scorebo_notrunc_MC': 'orange',
    'ScoreBO_M': 'limegreen',
    'JES-e-LB2': '#377eb8',
    'JES-e-LB2ben': 'goldenrod',
    'JES': 'goldenrod',
    'JES_2': 'dodgerblue',
    'JES-LB2': '#377eb8',
    'JES-e': 'navy',
    'JES-FB': 'crimson',
    'nJES-e': 'limegreen',
    'JESy': 'goldenrod',
    'JESy-e': 'darkgoldenrod',
    'MES': '#4daf4a',
    'EI-pi': '#e41a1c',
    'NEI-pi': 'deeppink',
    'EI': '#f781bf',
    'KG': 'crimson',
    'VES': 'dodgerblue',
    'NEI': 'k',
    'Sampling': 'orange',
    # BETA TESTS

    'MCpi-EI': 'grey',
    'MCpi-EI_notrans': 'navy',
    'MCpi-KG': 'forestgreen',

    'UCB': 'orangered',
    'MCpi-UCB': 'lightsalmon',
    'WAAL': '#e41a1c',
    'GIBBON': '#4daf4a',
    'WAAL-f': '#e41a1c',
    'BALD': '#ff7f00',
    'BALM': '#ff7f00',
    'QBMGP': '#377eb8',
    'BQBC': '#4daf4a',
    'ScoreBO_S01': 'crimson',
    'ScoreBO_S001': 'yellow',
    'ScoreBO_S0': 'orange',
    'noisy_ScoreBO_S01': 'brown',
    'ScoreBO_S': 'deeppink',

    'ScoreBO_J_HR': '#e41a1c',
    'ScoreBO_J_HR_MC': '#e41a1c',
    'BOTorch_mean_prior': '#4daf4a',
    'Bad_prior': '#984ea3',
    'ALBO_prior': '#377eb8',
    'correct': '#ff7f00',
    'ScoreBO_warm': 'orange',
    'ScoreBO_large': 'dodgerblue',
    'ScoreBO_J_HR_BOinit': 'dodgerblue',
    'ScoreBO_J_HR_ALinit': 'limegreen',
    'ScoreBO_J_HR_MC_v4': 'dodgerblue',
    'ScoreBO_J_HR_MC_v4_512': 'navy',
    'ScoreBO_J_HR_MMfix': '#e41a1c',
    'SAL_HR': 'navy',
    'SAL_WS': '#e41a1c',
    'NEI_MAP': 'orange',
    'NEI_no': 'dodgerblue',
    'SAL_WS_MC': 'dodgerblue',
    'ScoreBO_J_HR_notrunc_MC': 'dodgerblue',
    'ScoreBO_J_HR_notrunc_WS': 'navy',
    'NEI': 'k',
    'NEI_AL': 'darkgoldenrod',
    'JES-e-LB2_AL': 'navy',
    'JES_MAP': 'dodgerblue',
    'ScoreBO_J_HR_AL': 'purple',

    'cma_es': '#377eb8',
    'random_search': '#ff7f00',
    'hesbo20': '',
    'hesbo10': '#f781bf',
    'alebo_101_': '#a65628',
    'alebo_201_': '#984ea3',
    'turbo-1-b1': '#999999',
    'turbo-5-b1': '#e41a1c',
    'saasbo': '#dede00',
    'gtbo': 'k',
    'baxus': '#4daf4a',

    'gtbo_turbo:False_reuse:False_gt:300': '#ff7f00',
    'gtbo_turbo:False_reuse:True_gt:300': 'darkgoldenrod',

    'gtbo_turbo:True_reuse:False_gt:200': '#ff7f00',
    'gtbo_turbo:True_reuse:False_gt:300': 'darkgoldenrod',
    'gtbo_turbo:True_reuse:True:200': 'forestgreen',

}


init = {
    'active_hartmann6': 100,
    'stybtang10': 11,
    'gp_2_2_2_2dim': 3,
    'gp_2_2_2dim': 5,
    'gp_2_2dim': 3,
    'botorch_3_3_2dim': 0,
    'branin': 3,
    'branin_25': 3,
    'hartmann3': 4,
    'hartmann6': 7,
    'hartmann6_25': 5,
    'ackley8': 9,
    'ackley4': 5,
    'lasso_dna': 5,
    'ackley5': 6,
    'ackley4_25': 5,
    'alpine5': 6,
    'rosenbrock4': 5,
    'rosenbrock4_25': 5,
    'rosenbrock8': 9,
    'rosenbrock12': 13,
    'mich5': 6,
    'levy12': 13,
    'xgboost': 9,
    'fcnet': 7,
    'gp_2dim': 3,
    'gp_4dim': 5,
    'gp_6dim': 7,
    'gp_8dim': 9,
    'gp_12dim': 13,
    'active_branin': 3,
    'gramacy1': 2,
    'gramacy2': 3,
    'higdon': 2,
    'ishigami': 4,
    'hartmann4': 5,
    'rosenbrock2': 3,
    'hpo_blood': 5,
    'hpo_segment': 5,
    'hpo_australian': 5,
    'botorch_3_3_4dim': 5,
    'ackley12': 13,
    'gp_2_2_2_2_2dim': 5,
    'gp_1_1_4_4dim': 5,
    'gp_1_1_2_4_8dim': 5,
    'gramacy2': 3,
    'gramacy1': 3,
    'higdon': 3,
    'ishigami': 3,
    'active_hartmann6': 3,
    'active_branin': 3,
    'cosmo': 3,
    'rosenbrock20': 20 + 1,
    'rastrigin10': 10 + 1,
    'rastrigin20': 20 + 1,
    'levy10': 10 + 1,
    'levy16': 16 + 1,
    'hartmann12': 12 + 1,
    'ackley10': 10 + 1,
    'ackley16': 16 + 1,
    'hartmann6': 10,
    'levy4': 10,
    'branin2': 10,
    'lasso-dna': 10,
    'mopta08': 10,
    'svm': 10,

}


NAMES = {
    'SAL_HR': 'SAL - HR',
    'SAL_WS': 'SAL - WS',
    'JES_ben': 'JES-$y$',
    'ScoreBO_J': 'ScoreBO_J',
    'ScoreBO_M': 'ScoreBO_M',
    'GIBBON': 'GIBBON',
    'JES': 'JES',
    'JES-e': '$f$-JES-\u03B3',
    'JES-FB': 'FB-JES',
    'JES-e-LB2': 'JES',
    'JES-e-LB2AL': 'JES - AL',
    'JES-LB2': 'LB2-JES',
    'nJES': 'newJES',
    'nJES-e': 'newJES-\u03B3',
    'JESy': '$y$-JES',
    'JESy-e': '$y$-JES-\u03B3',
    'JES-e-pi': '\u03B3-JES-pi',
    'MES': 'MES',
    'EI': 'EI',
    'NEI': 'EI',
    'JES-pi': 'JES-pi',
    'EI-pi': 'EI-pi',
    'NEI-pi': 'Noisy EI-pi',
    'JES-e01': '\u03B3-0.1-JES',
    'JES-e01-pi': '\u03B3-0.1-JES-pi',
    'Sampling': 'Prior Sampling',
    # BETA TESTS
    'JES-e-pi-2': '\u03B3-JES-pi-2',
    'JES-e-pi-5': '\u03B3-JES-pi-5',
    'JES-e-pi-20': '\u03B3-JES-pi-20',
    'JES-e-pi-50': '\u03B3-JES-pi-50',
    'VES': 'VES',
    'KG': 'KG',
    'NEI_MAP': 'EI - MAP',
    'NEI_AL': 'Noisy EI - AL init',
    'NEI_no': 'EI - sobol init',
    'NEI_temp': 'EI',
    'NEI_correct': 'Correct HPs',
    'NEI_correct_name': 'EI',

    'NEI_wide': 'Wide LogNormal Prior',
    'NEI_botorch': 'BoTorch Prior',

    'WAAL': 'SAL-WS',
    'WAAL-f': 'SAL-WS',
    'BALD': 'BALD',
    'BALM': 'BALM',
    'QBMGP': 'QBMGP',
    'BQBC': 'BQBC',
    'ScoreBO_M_HR': 'ScoreBO_M - HR',
    'ScoreBO_M_BC': 'ScoreBO_M - BC',
    'ScoreBO_M_JS': 'ScoreBO_M - JS',
    'ScoreBO_M_WS': 'ScoreBO_M - WS',
    'ScoreBO_M_HR_star': 'ScoreBO*_M - HR',
    'ScoreBO_M_BC_star': 'ScoreBO*_M - BC',
    'ScoreBO_M_JS_star': 'ScoreBO*_M - JS',
    'ScoreBO_M_WS_star': 'ScoreBO*_M - WS',
    'ScoreBO_J_HR': 'SCoreBO',
    'ScoreBO_J_HR_notrunc': 'SCoreBO',
    'ScoreBO_J_HR_notrunc_MC': 'SCoreBO - HR',
    'ScoreBO_J_HR_notrunc_WS': 'SCoreBO - WS',
    'Scorebo_notrunc_MC': 'SCoreBO - MC',
    'ScoreBO_J_BC': 'ScoreBO_J - BC',
    'ScoreBO_J_JS': 'ScoreBO_J - JS',
    'ScoreBO_J_WS': 'ScoreBO_J - WS',
    'ScoreBO_J_HR_star': 'ScoreBO*_J - HR',
    'ScoreBO_J_BC_star': 'ScoreBO*_J - BC',
    'ScoreBO_J_JS_star': 'ScoreBO*_J - JS',
    'ScoreBO_J_WS_star': 'ScoreBO*_J - WS',

    'JES_2': 'JES_2',
    'ScoreBO_J_noinit_HR': 'ScoreBO_J - HR',
    'Bad_prior': 'Bad prior',
    'BOTorch_mean_prior': 'BOTorch prior',
    'ALBO_prior': 'Good prior',
    'correct': 'Correct HPs',
    'ScoreBO_warm': 'ScoreBO warmup256',
    'ScoreBO_J_HR_init': 'ScoreBO AL init',
    'ScoreBO_large': 'brown',
    'ScoreBO_J_HR_BOinit': 'ScoreBO - Full init',
    'ScoreBO_J_HR_ALinit': 'ScoreBO - AL init',
    'ScoreBO_J_HR_MC_v4': 'ScoreBO - MC (128)',
    'ScoreBO_J_HR_MC_v4_512': 'ScoreBO - MC (512)',
    'ScoreBO_J_HR_MC': 'SCoreBO - HR',
    'ScoreBO_J_HR_MMfix': 'ScoreBO - Moment match',
    'SAL_WS_MC': 'SAL - MC',
    'JES-e-LB2ben': 'JES - $y*$',



    # GTBO stuff:
    'cma_es': 'CMA-ES',
    'random_search': 'Random Search',
    'baxus': 'BaXUS',
    'hesbo10': 'HeSBO-10',
    'alebo_101_': 'ALEBO-10',
    'alebo_201_': 'ALEBO-20',
    'turbo-1-b1': 'TuRBO-1',
    'turbo-5-b1': 'TuRBO-5',
    'saasbo': 'SAASBO',
    'gtbo': 'GTBO',

    'gtbo_turbo:False_reuse:False_gt:300': 'GTBO',
    'gtbo_turbo:False_reuse:True_gt:300': 'GTBO-Re',


    'gtbo_turbo:True_reuse:False_gt:200': 'GTBO-200',
    'gtbo_turbo:True_reuse:False_gt:300': 'GTBO-300',
    'gtbo_turbo:True_reuse:True:200': 'GTBO-Re-200',



}

PLOT_LAYOUT = dict(
    linewidth=2,
    markevery=20,
    markersize=0,
    markeredgewidth=4,
    marker='*'
)

BENCHMARK_NAMES = {
    'hartmann3': 'Hartmann-3',
    'hartmann4': 'Hartmann-4',
    'hartmann6': 'Hartmann-6',
    'hartmann6_25': 'Hartmann-6 (25D)',
    'rosenbrock2': 'Rosenbrock-2',
    'rosenbrock4': 'Rosenbrock-4',
    'rosenbrock8': 'Rosenbrock-8',
    'rosenbrock12': 'Rosenbrock-12',
    'ackley8': 'Ackley-8',
    'ackley12': 'Ackley-12',
    'ackley16': 'Ackley-16',
    'ackley4': 'Ackley-5',
    'ackley5': 'Ackley-5',
    'levy12': 'Levy-12',
    'levy16': 'Levy-16',
    'rosenbrock4_25': 'Rosenbrock-4 / 25D',
    'branin': 'Branin',
    'branin_25': 'Branin / 25D',
    'ackley4_25': 'Ackley-4 (25D)',
    'gp_4dim': 'GP-sample (4D)',
    'gp_8dim': 'GP-sample (8D)',
    'gp_12dim': 'GP-sample (12D)',
    'hpo_blood': 'hpo_blood',
    'hpo_segment': 'hpo_segment',
    'hpo_australian': 'hpo_australian',
    'gp_2_2dim': 'GP-sample (2D+2D)',
    'botorch_3_3_2dim': 'GP-sample (3D+3D+2D)',
    'botorch_3_3_4dim': 'GP-sample (3D+3D+4D)',
    'gp_2_2_2_2dim': 'GP-sample (2D+2D+2D+2D)',
    'gp_2_2_2_2_2dim': 'GP-sample (2D+2D+2D+2D+2D)',
    'gp_2_2_2dim': 'GP-sample (2D+2D+2D)',
    'gp_1_1_4_4dim': 'GP-sample (1D+1D+4D+4D)',
    'gp_1_2_3_4dim': 'GP-sample (1D+2D+3D+4D)',
    'lasso_dna': 'Lasso-DNA (180D)',
    'gramacy2': 'Gramacy (2D)',
    'gramacy1': 'Gramacy (1D)',
    'higdon': 'Higdon (1D)',
    'mich5': 'Mich-5',
    'ishigami': 'Ishigami',
    'active_hartmann6': 'Hartmann-6',
    'active_branin': 'Branin',
    'gp_1_1_2_4_8dim': 'GP(1D+1D+2D+2D+4D+8D',
    'cosmo': 'Cosmological Constants (11D)',
    'rosenbrock20': 'Rosenbrock-20',
    'rastrigin10': 'Rastrigin-10',
    'rastrigin20': 'Rastrigin-20',
    'levy10': 'Levy-10',
    'hartmann12': 'Hartmann-12',
    'ackley10': 'Ackley-10',
    'stybtang10': 'Styblinski-Tang-10',
    'levy4': 'Levy (4D) - 300D Embedding',
    'branin2': 'Branin (2D) - 300D Embedding',
    'lasso-dna': 'Lasso-DNA',
    'mopta08': 'MOPTA08',
    'svm': 'SVM,'

}


def get_gp_regret(gp_task_type):
    gp_task_files = glob(join(dirname(abspath(__file__)),
                              f'gp_sample/{gp_task_type}/*.json'))
    opts = {}
    for file in sorted(gp_task_files):
        name = file.split('/')[-1].split('_')[-1].split('.')[0]
        with open(file, 'r') as f:
            opt_dict = json.load(f)
            opts[name] = opt_dict['opt']
    return opts


def get_regret(benchmark):
    if 'gp_' in benchmark:
        gp_regrets = get_gp_regret(benchmark)
        for key in gp_regrets.keys():
            gp_regrets[key] = gp_regrets[key] + 0.15
        return gp_regrets

    else:
        regrets = {
            'gp_12dim': 0,
            'branin2': -0.397887,

            'branin': -0.397887,
            'branin_25': -0.397887,
            'hartmann3': 3.86278,
            'hartmann6': 3.32237,
            'hartmann6_25': 3.32237,
            'hartmann4': 3.1344945430755615,
            'ackley4': 0,
            'ackley5': 0,
            'ackley4_25': 0,
            'ackley8': 0,
            'ackley12': 0,
            'alpine5': 0,
            'rosenbrock12': 0,
            'rosenbrock4': 0,
            'rosenbrock4_25': 0,
            'rosenbrock2': 0,
            'rosenbrock8': 0,
            'gp_2dim': 0,
            'gp_4dim': 0,
            'gp_6dim': 0,
            'levy12': 0,
            'fcnet': 0.03 + np.exp(-5) + np.exp(-6.6) ,
            'xgboost': -(8.98 + 2 * np.exp(-6)),

            'active_branin': 0,
            'gramacy1': 0,
            'gramacy2': -0.428882,
            'higdon': 0,
            'ishigami': 10.740093895930428 + 1e-3,
            'active_hartmann6': 3.32237,
            'gp_2_2dim': 0,
            'botorch_3_3_2dim': 0,
            'botorch_2_2_2_2dim': 0,
            'botorch_3_3_4dim': 0,
            'gp_2_2_2_2dim': 0,
            'gp_2_2_2dim': 0,
            'lasso_dna': None,
            'mich5': 4.687658,
            'gp_2_2_2_2_2dim': 0,
            'gp_1_1_2_4_8dim': 5,
            'rosenbrock20': 0,
            'rastrigin10': 0,
            'rastrigin20': 0,
            'stybtang10': -39.16599 * 10,
            'levy10': 0,
            'levy16': 0,
            'hartmann12': 2 * 3.32237,
            'ackley10': 0,
            'ackley16': 0
        }
        return regrets.get(benchmark, False)


def process_funcs_args_kwargs(input_tuple):
    '''
    helper function for preprocessing to assure that the format of (func, args, kwargs is correct)
    '''
    if len(input_tuple) != 3:
        raise ValueError(
            f'Expected 3 elements (callable, list, dict), got {len(input_tuple)}')

    if not callable(input_tuple[0]):
        raise ValueError('Preprocessing function is not callable.')

    if type(input_tuple[1]) is not list:
        raise ValueError('Second argument to preprocessing function is not a list.')

    if type(input_tuple[2]) is not dict:
        raise ValueError('Third argument to preprocessing function is not a dict.')

    return input_tuple


def filter_paths(all_paths, included_names=None):
    all_names = [benchmark_path.split('/')[-1]
                 for benchmark_path in all_paths]
    if included_names is not None:
        used_paths = []
        used_names = []

        for path, name in zip(all_paths, all_names):
            if name in included_names:
                used_paths.append(path)
                used_names.append(name)
        return used_paths, used_names

    return all_paths, all_names


def get_files_from_experiment(experiment_name, benchmarks=None, acquisitions=None):
    '''
    For a specific expefiment, gets a dictionary of all the {benchmark: {method: [output_file_paths]}}
    as a dict, includiong all benchmarks and acquisition functions unless specified otherwise in
    the arguments.
    '''
    paths_dict = {}
    all_benchmark_paths = glob(join(experiment_name, '*'))
    # print(join(experiment_name, '*'))
    filtered_benchmark_paths, filtered_benchmark_names = filter_paths(
        all_benchmark_paths, benchmarks)

    # *ensures hidden files are not included
    for benchmark_path, benchmark_name in zip(filtered_benchmark_paths, filtered_benchmark_names):
        paths_dict[benchmark_name] = {}
        all_acq_paths = glob(join(benchmark_path, '*'))
        filtered_acq_paths, filtered_acq_names = filter_paths(
            all_acq_paths, acquisitions)

        for acq_path, acq_name in zip(filtered_acq_paths, filtered_acq_names):
            run_paths = glob(join(acq_path, '*[0-9].csv'))
            paths_dict[benchmark_name][acq_name] = sorted(run_paths)

    return paths_dict


def get_dataframe(paths, funcs_args_kwargs=None, idx=0):
    '''
    For a given benchmark and acquisition function (i.e. the relevant list of paths),
    creates the dataframe that includes the relevant metrics.

    Parameters:
        paths: The paths to the experiments that should be included in the dataframe
        funcs_args_kwargs: List of tuples of preprocessing arguments,
    '''
    # ensure we grab the name from the right spot in the file structure
    names = [path.split('/')[-1].split('.')[0] for path in paths]

    # just create the dataframe and set the column names
    complete_df = pd.DataFrame(columns=names)

    # tracks the maximum possible length of the dataframe
    max_length = None

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for path, name in zip(paths, names):
            # print(path)
            per_run_df = pd.read_csv(path)
            # this is where we get either the predictions or the true values
            if funcs_args_kwargs is not None:
                for func_arg_kwarg in funcs_args_kwargs:
                    func, args, kwargs = process_funcs_args_kwargs(func_arg_kwarg)
                    # try:
                    per_run_df = func(per_run_df, name, *args, **kwargs)
                    # except:
                    #    print('unable to retrieve',
                    #     path)
            complete_df.loc[:, name] = per_run_df.iloc[:, 0]
    return complete_df


def get_min(df, run_name, metric, minimize=True):
    if 'gp_' in run_name or 'lasso' in run_name:
        minimize = False
    min_observed = np.inf
    mins = np.zeros(len(df))
    for r, row in enumerate(df[metric]):
        if minimize:
            if (row < min_observed) and row != 0:
                min_observed = row
            mins[r] = min_observed
        else:
            if -row < min_observed and row != 0:
                min_observed = -row
            mins[r] = min_observed
    return pd.DataFrame(mins, columns=[run_name])


def get_metric(df, run_name, metric, minimize=True):
    nonzero_elems = df[metric][df[metric] != 0].to_numpy()
    first_nonzero = nonzero_elems[0]
    num_to_append = np.sum(df[metric] == 0)
    result = np.append(np.ones(num_to_append) * first_nonzero, nonzero_elems)

    if metric == 'RMSE':
        result = np.log10(result)
    return pd.DataFrame(result, columns=[run_name])


def compute_regret(df, run_name, regret, log=True):
    if type(regret) is dict:
        run_name_short = ''.join(run_name.split('_')[-2:])
        regret = regret[run_name_short]

    if np.any(df.iloc[:, 0] + regret) < 0:
        vals = df.iloc[:, 0]
        error_msg = f'Regret value: {regret}, best observed {-vals[vals + regret < 0]} for run {run_name_short }.'
        f'Re-optimize GP.'
        raise ValueError(error_msg)

    if log:
        mins = df.iloc[:, 0].apply(lambda x: np.log10(x + regret))
    else:
        mins = df.iloc[:, 0].apply(lambda x: x + regret)
    return pd.DataFrame(mins)


def compute_nothing(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: x)
    else:
        mins = df.iloc[:, 0].apply(lambda x: x)
    return pd.DataFrame(mins)


def compute_negative(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: x)
    else:
        mins = df.iloc[:, 0].apply(lambda x: x)

    return pd.DataFrame(-mins)


def get_empirical_regret(data_dicts, metric, idx=0, log_coeff=0.13):
    MAX_REGRET = -1e10
    regrets = None
    for data_dict in [data_dicts]:

        for run_name, files in data_dict.items():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                names = [f'run' + file.split('/')[-1].split('_')
                         [-1].split('.')[0] for file in files]

                # just create the dataframe and set the column names
                if regrets is None:
                    regrets = {f'{name}': MAX_REGRET for name in names}
                for path, name in zip(files, names):
                    per_run_df = pd.read_csv(path).loc[:, metric]
                    # this is where we get either the predictions or the true values

                    if per_run_df.max() > regrets[name]:
                        regrets[name] = -per_run_df.max()

    return {name: -regret + log_coeff for name, regret in regrets.items()}


def compute_significance(reference, data_dict):
    from scipy.stats import ttest_ind
    for benchmark in data_dict.keys():

        reference_data = data_dict[benchmark].pop(reference, None)
        if reference is None:
            print('The provided reference', reference, 'does not exist in the data array.'
                  'Available runs are', list(data_dict.keys()), '. Cannot compute significance.')
            return
        ref_stats = reference_data[:, -1]
        for comp, vals in data_dict[benchmark].items():
            comp_stats = vals[:, -1]
            res = ttest_ind(ref_stats, comp_stats, alternative='less')
            print(benchmark, reference, comp, res)


def compute_ranking(all_data, ax, plot_config):
    # num_steps x num_runs x num_methods
    maxlen = 0
    ranking_data = {}
    for benchmark in all_data.keys():
        print(benchmark)
        num_acqs = len(all_data[benchmark])
        benchmark_lengths = all_data[benchmark][list(
            all_data[benchmark].keys())[0]].shape[0]
        num_runs = all_data[benchmark][list(all_data[benchmark].keys())[0]].shape[1]
        benchmark_data = np.empty((benchmark_lengths, num_runs, num_acqs))

        for i, (acq, data) in enumerate(all_data[benchmark].items()):

            print(acq)
            benchmark_data[..., i] = data

        # ranking = benchmark_data.mean(1).argsort(axis=-1)
        ranking = benchmark_data.argsort(axis=-1)
        maxlen = max(maxlen, benchmark_lengths)
        ranking_data[benchmark] = ranking

    num_benchmarks = len(all_data)
    num_runs = 25
    extended_rankings = np.zeros((maxlen, num_benchmarks * num_runs, num_acqs))
    for bench_idx, bench_data in enumerate(ranking_data.values()):
        print(bench_idx, bench_data.shape)
        wrong_ratio = len(bench_data) / maxlen
        for i in range(maxlen):
            # print(bench_idx, i, i * wrong_ratio, ben   ch_data.shape)
            # extended_rankings[i, bench_idx, :] = bench_data[int(i * wrong_ratio), :]
            extended_rankings[i, bench_idx
                              * num_runs: (bench_idx + 1) * num_runs, :] = bench_data[int(i * wrong_ratio), :]
    acq_order = all_data[benchmark].keys()
    for idx, acq in enumerate(acq_order):
        if acq == 'GIBBON':
            acq_sub = 'JES-e-LB2'
        elif acq == 'JES-e-LB2':
            acq_sub = 'GIBBON'

        elif acq == 'BQBC':
            acq_sub = 'BALM'
        elif acq == 'BALM':
            acq_sub = 'BQBC'
        else:
            acq_sub = acq
        acq_ranks = extended_rankings.mean(1)[..., idx]
        MA = 10
        acq_ranks = np.convolve(acq_ranks, np.ones(MA), 'valid') / MA
        ax.plot(np.linspace(MA, 100, maxlen - MA + 1),
                acq_ranks + 1, label=NAMES[acq_sub], color=COLORS[acq_sub], linewidth=2)

    # for idx, acq in enumerate(acq_order):
    #    plt.plot(np.linspace(0, 100, len(
    #        ranking_data['hartmann6'])), ranking_data['hartmann6'][:, 20:30][..., idx], label=NAMES[acq], color=COLORS[acq])
        # ax.legend(fontsize=plot_config['fontsizes']['legend'])
    #ax.set_xlabel('Percentage of run', fontsize=plot_config['fontsizes']['metric'])
    ax.tick_params(axis='x', labelsize=13)

    import matplotlib.ticker as mtick
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Percentage of run', fontsize=plot_config['fontsizes']['metric'])
    ax.tick_params(axis='y', labelsize=13)
    ax.set_title('Relative ranking',
                 fontsize=plot_config['fontsizes']['benchmark_title'])


SIZE_CONFIGS = {
    # GTBO:
    'gtbo_syn':
    {
        'reference_run': 'SAASBO',
        'subplots':
        {
            'figsize': (20, 5.1),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 21,
            'metric': 18,
            'iteration': 15,
            'legend': 17,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Simple Regret',
        'metric': 'f',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': True,
    },
        'gtbo_real':
    {
        'reference_run': 'SAASBO',
        'subplots':
        {
            'figsize': (20, 5.1),
            'nrows': 1,
            'ncols': 2,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 21,
            'metric': 18,
            'iteration': 15,
            'legend': 17,
        },
        'plot_args':
        {
        },
        'metric_name': 'Best observed value',
        'metric': 'f',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': False,
    },



    'synthetic_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (20, 4.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'Guess values',
        'compute': compute_regret,
        'get_whatever': get_metric,
        'log_regret': True,

    },
    'synthetic_al':
    {
        'reference_run': 'SAL_HR',
        'subplots':
        {
            'figsize': (20, 4.5),
            'nrows': 1,
            'ncols': 7,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
            'infer_ylim': True,
            'start_at': 20
        },
        'metric_name': 'Negative MLL',
        'metric': 'MLL',
        'compute': compute_nothing,
        'get_whatever': get_metric,
        'log_regret': False,

    },
    'synthetic_al_rmse':
    {
        'reference_run': 'SAL_HR',
        'subplots':
        {
            'figsize': (20, 4.5),
            'nrows': 1,
            'ncols': 7,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log RMSE',
        'metric': 'RMSE',
        'compute': compute_nothing,
        'get_whatever': get_metric,
        'log_regret': False,
            'legend': 15,

    },
    'gp_prior':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (7, 4),
            'nrows': 1,
            'ncols': 1,
        },
        'tight_layout':
        {
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 0,
            'legend': 16,
        },
        'plot_args':
        {
        },
        'metric_name': 'Simple Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': False,

    },
    'saasbo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (20, 3.6),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.75,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 0,
            'legend': 16,
        },
        'plot_args':
        {
            'start_at': 1,
            'init': 3,
        },
        'metric_name': 'Min. Observed Value',
        'metric': ('ackley4_25', 'hartmann6_25', 'lasso_dna'),
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': False,


    },
    'addgp_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (15, 4.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.75,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 0,
            'legend': 16,
        },
        'plot_args':
        {
        },
        'metric_name': ('Simple Regret', 'Simple Regret', 'Min. Observed Value'),
        'metric': 'True Eval',
        'compute': (compute_regret, compute_regret, compute_nothing),
        'get_whatever': get_min,
        'log_regret': False,
        'empirical_regret': True,
    },


}


def plot_optimization(data_dict,
                      preprocessing=None,
                      title='benchmark',
                      path_name='',
                      linestyle='solid',
                      xlabel='X',
                      ylabel='Y',
                      use_empirical_regret=False,
                      fix_range=None,
                      start_at=0,
                      only_plot=-1,
                      names=None,
                      predictions=False,
                      init=2,
                      n_markers=20,
                      n_std=1,
                      show_ylabel=True,
                      maxlen=0,
                      plot_ax=None,
                      show_xlabel=True,
                      first=True,
                      show_noise=None,
                      infer_ylim=False,
                      lower_bound=None,
                      hide=False,
                      plot_config=None,
                      custom_order=None,
                      no_legend=False,
                      ):
    lowest_doe_samples = 1e10
    results = {}
    if plot_ax is None:
        fig, ax = plt.subplots(figsize=(25, 16))
    else:
        ax = plot_ax
    ax_max = -np.inf
    ax_min = np.inf
    if use_empirical_regret:
        emp_regrets = get_empirical_regret(
            data_dict, metric=preprocessing[0][2]['metric'])
        for step_ in preprocessing:
            if step_[0] is compute_regret:
                step_[2]['regret'].update(emp_regrets)
    else:
        emp_regrets = None

    min_ = np.inf
    benchmark_data = {}
    for run_name, files in data_dict.items():
        plot_layout = copy(PLOT_LAYOUT)

        if hide:
            plot_layout['c'] = 'white'
            plot_layout['label'] = '__nolabel__'
        else:
            plot_layout['linestyle'] = linestyle
            plot_layout['c'] = COLORS.get(run_name, 'k')
            plot_layout['label'] = NAMES.get(run_name, 'Nameless Run') + ' ' + path_name
            if plot_layout['label'] == 'Nameless Run':
                continue
        if no_legend:
            plot_layout['label'] = '__nolabel__'
        result_dataframe = get_dataframe(files, preprocessing)
        # convert to array and plot

        data_array = result_dataframe.to_numpy()
        if only_plot > 0:
            data_array = data_array[:, 0:only_plot]

        only_complete = True
        if only_complete:
            complete_mask = ~np.any(np.isnan(data_array), axis=0)
            data_array = data_array[:, complete_mask]

        benchmark_data[run_name] = data_array

        y_mean = data_array.mean(axis=1)
        y_std = sem(data_array, axis=1)
        markevery = np.floor(len(y_mean) / n_markers).astype(int)
        plot_layout['markevery'] = markevery

        if maxlen:
            y_mean = y_mean[0:maxlen]
            y_std = y_std[0:maxlen]
            X = np.arange(0 + 1, maxlen + 1)

        else:
            X = np.arange(1, len(y_mean) + 1)

        y_max = (y_mean + y_std)[start_at:].max()
        y_min = (y_mean - y_std)[start_at:].min()

        clrs = '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', 'k', 'navy', 'darkgoldenrod'
        # for i, clr in enumerate(clrs):
        #    lo = copy(plot_layout)
        #    lo['c'] = clr
        #    if run_name == 'NEI':
        #        lo['linestyle'] = ':'
        #    try:
        #        ax.plot(X[start_at:], data_array[:, i][start_at:], **lo)
        #    except:
        #        pass
        if start_at > 0 and not infer_ylim:
            X = X[start_at:]
            y_mean = y_mean[start_at:]
            y_std = y_std[start_at:]

        ax.plot(X, y_mean, **plot_layout)
        ax.fill_between(X, y_mean - n_std * y_std, y_mean + n_std
                        * y_std, alpha=0.1, color=plot_layout['c'])
        ax.plot(X, y_mean - n_std * y_std, alpha=0.5, color=plot_layout['c'])
        ax.plot(X, y_mean + n_std * y_std, alpha=0.5, color=plot_layout['c'])
        min_ = min((y_mean - n_std * y_std).min(), min_)

        ax_max = np.max([y_max, ax_max])
        ax_min = np.min([y_min, ax_min])
    if fix_range is not None:
        ax.set_ylim(fix_range)
    elif infer_ylim:
        diff_frac = (ax_max - ax_min) / 20
        ax.set_ylim([ax_min - diff_frac, ax_max + diff_frac])
    ax.axvline(x=init, color='k', linestyle=':', linewidth=2)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if lower_bound is not None:
        ax.axhline(y=lower_bound, color='k', linestyle='--', linewidth=2)

    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=plot_config['fontsizes']['metric'])
    ax.set_title(title, fontsize=plot_config['fontsizes']['benchmark_title'])
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=plot_config['fontsizes']['metric'])
    if first and not hide:
        handles, labels = ax.get_legend_handles_labels()
        sorted_indices = np.argsort(labels[:-1])
        sorted_indices = np.append(sorted_indices, len(labels) - 1)
        if custom_order is not None:
            try:
                ax.legend(np.array(handles)[custom_order],
                          np.array(labels)[custom_order], fontsize=plot_config['fontsizes']['legend'])
            except:
                pass
        else:
            ax.legend(np.array(handles)[sorted_indices],
                      np.array(labels)[sorted_indices], fontsize=plot_config['fontsizes']['legend'])

    ax.grid(visible=True)
    return benchmark_data


if __name__ == '__main__':
    acqs = [
        # 'ScoreBO_J_HR_notrunc',
        'ScoreBO_J_HR_notrunc_WS',
        'ScoreBO_J_HR_MC',
        # 'ScoreBO_J_HR',
        # 'JES-e-LB2',
        # 'GIBBON',
        # 'NEI',
        # 'cma_es',
        # 'random_search',
        # 'baxus',
        # 'hesbo10',
        # 'alebo_101_',
        # 'alebo_201_',
        # 'turbo-1-b1',
        # 'turbo-5-b1',
        # 'saasbo',
        # 'gtbo',
        # 'gtbo_turbo:False_reuse:False_gt:300',
        # 'gtbo_turbo:False_reuse:True_gt:300',


        # 'gtbo_turbo:True_reuse:False_gt:200',
        # 'gtbo_turbo:True_reuse:False_gt:300',
        # 'gtbo_turbo:True_reuse:True:200',
    ]

    all_data = {}
    config = 'synthetic_al_rmse'
    plot_config = SIZE_CONFIGS[config]
    # acqs = 'SAL_WS', 'SAL_HR'
    #
    #
    acqs = 'SAL_WS', 'SAL_HR', 'BQBC', 'QBMGP', 'BALM'
    include_ranking = False

    benchmarks = BENCHMARK_PACKS[config]['names']

    lower_bounds = BENCHMARK_PACKS[config].get('best', [None] * len(benchmarks))
    get_whatever = get_metric
    num_benchmarks = len(benchmarks)
    if num_benchmarks == 0:
        raise ValueError('No files')

    num_rows = plot_config['subplots']['nrows']
    cols = plot_config['subplots']['ncols']

    empirical_regret_flag = plot_config.get('empirical_regret', False)
    if include_ranking:
        plot_config['subplots']['ncols'] += 1
    fig, axes = plt.subplots(**plot_config['subplots'])

    paths_to_exp = [
        # 'results/20230502_al_with_outputscale',
        # 'results/20230308_gp_prior',
        # 'results/20230308_correct_n1',
        # 'results/20230308_wide_n1',
        # 'results/20230308_botorch_n1',
        # 'results/20230413_ScoreBO_comp_head-acqmaxiter200',
        # 'results/20230414_ScoreBO_new',
        # 'results/20230420_addGP_rbf05',
        # 'results/20230511_ScoreBO_RFFLarge',
        # 'results/20230414_ScoreBO_new',
        # 'results/20230512_cosmo'
        # 'results/20230507_ScoreBO_SAAS'
        # 'results/20230509_ScoreBO_SAAS'
        # 'results/gtbo'
        'results/20230502_al_with_outputscale'
    ]

    # paths_to_exp = [
    #    'results/20230314_saasbo',
    # ]
    plot_metric = ['Simple Regret', 'Simple Regret', 'Missclass. Rate']
    # paths_to_exp = [
    #   'results/20230314_saasbo'
    # ]
    bench_len = {'gp_2_2_2dim': 60, 'gp_2_2_2_2_2dim': 99, 'lasso_dna': 150}
    # paths_to_exp = [
    # 'results/20230331_kg_bench',
    # 'results/20230415_AddGPFixnoise_nonoise'
    # ]
    linestyles = ['solid', 'solid', 'dashed']
    hide = [False, False, False, True]
    path_name = ['', '', '(MC)', '- Correct HPs', '- LogNormal Prior', '- BoTorch Prior',
                 '- Broad lognormal prior', '']

    all_files_list = [get_files_from_experiment(
        path_, benchmarks, acqs) for path_ in paths_to_exp]
    # raise SystemExit

    for path_idx, path_to_exp in enumerate(paths_to_exp):
        files = get_files_from_experiment(
            path_to_exp, benchmarks, acqs)
        # files = get_files_from_experiment(
        #    'results/20221231_final_al',
        #    ['gramacy1', 'higdon', 'gramacy2', 'active_branin',
        #        'ishigami', 'active_hartmann6'], acqs
        # )
        files = {bm: files[bm] for bm in benchmarks}
        for benchmark_idx, (benchmark_name, paths) in enumerate(files.items()):
            if isinstance(plot_config['compute'], tuple):
                compute_type = plot_config['compute'][benchmark_idx]
            else:
                compute_type = plot_config['compute']

            if isinstance(plot_config['metric_name'], tuple):
                metric_name = plot_config['metric_name'][benchmark_idx]
            else:
                metric_name = plot_config['metric_name']

            if isinstance(plot_config['metric'], tuple):
                metric = plot_config['metric'][benchmark_idx]
            else:
                metric = plot_config['metric']

            regret = get_regret(benchmark_name)
            # preprocessing = [(get_min, [], {'metric': 'Guess values'}), (compute_regret, [], {
            #    'log': True, 'regret': regrets[benchmark_name]})]
            # preprocessing = [(get_metric, [], {'metric': 'MLL'}), (compute_nothing, [], {
            #    'log': True, 'regret': regrets[benchmark_name]})]
            if num_benchmarks == 1:
                ax = axes

            elif num_rows == 1:
                ax = axes[benchmark_idx]
            else:
                ax = axes[int(benchmark_idx / cols), benchmark_idx % cols]

            preprocessing = [(plot_config['get_whatever'], [], {'metric': metric}), (compute_type, [], {
                'log': plot_config['log_regret'], 'regret': regret})]
            results = {}

            all_data[benchmark_name] = plot_optimization(paths,
                                                         xlabel='',
                                                         ylabel=metric_name,
                                                         n_std=0.5,
                                                         # plot_config['plot_args'].get('infer_ylim', False),
                                                         infer_ylim=False,
                                                         start_at=plot_config['plot_args'].get(
                                                             'start_at', init[benchmark_name] - 1) + 100 * int(benchmark_name == 'active_hartmann6') ,

                                                         # fix_range=(-1, 4),
                                                         preprocessing=preprocessing,
                                                         plot_ax=ax,
                                                         # benchmark_idx == len(files) - 1,
                                                         # ,
                                                         # benchmark_idx == len(files) - 1,
                                                         first=False,
                                                         n_markers=10,
                                                         show_ylabel=benchmark_idx % cols == 0,
                                                         init=plot_config['plot_args'].get(
                                                             'init', init[benchmark_name]),  # init[benchmark_name],
                                                         maxlen=bench_len.get(
                                                             benchmark_name, 0),
                                                         title=BENCHMARK_NAMES[benchmark_name],
                                                         use_empirical_regret=empirical_regret_flag and compute_type is compute_regret,
                                                         linestyle=linestyles[path_idx],
                                                         path_name=path_name[path_idx],
                                                         hide=False,
                                                         no_legend=True,
                                                         # no_legend=[
                                                         #    True, True, True, True, True, False][benchmark_idx],
                                                         show_xlabel=True,
                                                         lower_bound=lower_bounds[benchmark_idx],
                                                         plot_config=plot_config,
                                                         # custom_order=[0, 2, 4, 1, 3]
                                                         )
    from copy import deepcopy
    data_copy = deepcopy(all_data)

    # fig.suptitle('Synthetic', fontsize=36)

    # compute_significance(plot_config['reference_run'], data_copy)
    if True:
        compute_ranking(all_data, axes[-1], plot_config)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=len(acqs),
                   fontsize=plot_config['fontsizes']['legend'] + 3)
    else:
        plt.legend(fontsize=plot_config['fontsizes']['legend'] + 1)
    plt.tight_layout(**plot_config['tight_layout'])
    plt.savefig(f'neurips_plots/{config}ws.pdf', bbox_inches='tight')
    plt.show()
