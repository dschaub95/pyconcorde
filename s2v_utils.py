from TSP_utils import TSP_solver, TSP_plotter, TSP_generator, TSP_loader
import numpy as np
import networkx as nx
import tqdm
import tsplib95
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def get_cparams_from_lengths(cparam, delta= 0.005, num_nodes=50, num_samples=1000, data_root='test_sets'):
    # get lengths
    if cparam == 'random':
        folder = f'{data_root}/synthetic_n_{num_nodes}_{num_samples}/'
    else:
        folder = f'{data_root}/synthetic_n_{num_nodes}_cparam_{cparam}_delta_{delta}_{num_samples}/'
    with open(folder+'lengths.txt', 'r') as f:
        lines = f.readlines()
        file_names = [line.split(':')[0].strip() for k, line in enumerate(lines)]
        test_cparams = [float(line.split(':')[-1].strip()) / np.sqrt(1*50) for k, line in enumerate(lines)]
        cparam_dict = dict(zip(file_names, test_cparams))
    return cparam_dict

def get_soltime_from_file(cparam, delta=0.005, num_nodes=50, num_samples=1000, data_root='test_sets'):
    # get lengths
    if cparam == 'random':
        folder = f'{data_root}/synthetic_n_{num_nodes}_{num_samples}/'
    else:
        folder = f'{data_root}/synthetic_n_{num_nodes}_cparam_{cparam}_delta_{delta}_{num_samples}/'
    with open(folder+'sol_times.txt', 'r') as f:
        lines = f.readlines()
        file_names = [line.split(':')[0].strip() for k, line in enumerate(lines)]
        test_sol_times = [float(line.split(':')[-1].strip()) for k, line in enumerate(lines)]
        soltime_dict = dict(zip(file_names, test_sol_times))
    return soltime_dict

def get_soltimes_from_file(cparam, delta=0.005, num_nodes=50, num_samples=1000, data_root='test_sets'):
    # get lengths
    if cparam == 'random':
        folder = f'{data_root}/synthetic_n_{num_nodes}_{num_samples}/'
    else:
        folder = f'{data_root}/synthetic_n_{num_nodes}_cparam_{cparam}_delta_{delta}_{num_samples}/'
    with open(folder+'sol_times.txt', 'r') as f:
        lines = f.readlines()
        file_names = [line.split(':')[0].strip() for k, line in enumerate(lines)]
        test_sol_times = [[float(time.strip()) for time in line.split(':')[-1].split(',')] for k, line in enumerate(lines)]
        soltime_dict = dict(zip(file_names, test_sol_times))
    return soltime_dict

def get_approx_to_opt_from_file(approx_path='s2v_dqn_results/test_cparam_random.csv'):
    with open(approx_path, 'r') as f:
        lines = f.readlines()
        file_names = [line.split(',')[0].strip() for k, line in enumerate(lines)]
        approx_to_opt = [float(line.split(',')[1].strip()) for k, line in enumerate(lines)]
        approx_dict = dict(zip(file_names, approx_to_opt))
    return approx_dict

def get_approx_and_soltime_lists(approx_path, **args):
    approx_dict = get_approx_to_opt_from_file(approx_path)
    # soltime_dict = get_soltime_from_file(cparam)
    soltime_dict = get_soltimes_from_file(**args)
    approx_list = []
    soltime_list = []
    for key in approx_dict:
        approx_list.append(approx_dict[key])
        soltime_list.append(soltime_dict[key])
    return approx_list, soltime_list

def get_approx_and_cparam_lists(approx_path, **args):
    approx_dict = get_approx_to_opt_from_file(approx_path)
    cparam_dict = get_cparams_from_lengths(**args)
    approx_list = []
    cparam_list = []
    for key in approx_dict:
        approx_list.append(approx_dict[key])
        cparam_list.append(cparam_dict[key])
    return cparam_list, approx_list

def get_soltime_and_cparam_lists(**args):
    soltime_dict = get_soltimes_from_file(**args)
    cparam_dict = get_cparams_from_lengths(**args)
    soltime_list = []
    cparam_list = []
    for key in soltime_dict:
        soltime_list.append(soltime_dict[key])
        cparam_list.append(cparam_dict[key])
    return soltime_list, cparam_list

def get_reg_fit_data(x_list, y_list):
    X = np.array(x_list)
    X = sm.add_constant(X)
    Y = np.array(y_list)
    results = sm.OLS(Y, X).fit()
    b = results.params[0]
    a = results.params[1]
    x = np.arange(np.min(x_list), np.max(x_list), 0.0001)
    y = a * x + b
    # p_value = np.round(results.pvalues[1], 10)
    p_value = np.round(results.pvalues[1],30)
    rsquared = np.round(results.rsquared, 4)
    slope = np.round(a, 6)
    return x, y, p_value, rsquared, slope