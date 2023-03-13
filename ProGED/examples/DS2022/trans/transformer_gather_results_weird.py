
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.integrate import odeint
from src.generate_data.systems_collection import strogatz, mysystems
from plotting.parestim_sim_plots import plot_trajectories
import random

##
method = "transformer"
exp_type = 'sysident_num'

systems = {**strogatz, **mysystems}
data_version = 'all'
exp_version = 'e1'
structure_version = 's0'
analy_version = 'a4'
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}"
path_base_out = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{analy_version}{os.sep}"
os.makedirs(path_base_out, exist_ok=True)

sys_names = list(systems.keys())
snrs = ['inf', 30, 13]
set_obs = "full"
n_inits = 4
sim_step = 0.1
sim_time = 10
data_length = int(sim_time/sim_step)
plot_trajectories_bool = False

results_transformer = pd.read_csv(f"{path_base_in}ete{data_length}.csv")
results_transformer_clean_list = []

for icol in results_transformer.columns:
    icol = results_transformer_100.columns[100]
    trajectories_true = []
    trajectories_model = []
    trajectories_true_inf = []

    name_parts = icol.split('_')
    sys_name = name_parts[0]
    data_version = name_parts[1]
    snr = name_parts[3][3:]
    iinit = name_parts[4][-1]
    duration = float(results_transformer[icol][0])
    TE_first = float(results_transformer[icol][1])

    # extract expression
    if 'str' in str(type(results_transformer[icol][4])):
        expr_merged = [i for i in results_transformer[icol][2:5]]
    else:
        expr_merged = [i for i in results_transformer[icol][2:4]]
    num_expr = len(expr_merged)

    # load true data
    data_filepath = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
    data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
    itr_true = pd.read_csv(data_filepath + data_filename)
    itr_true = np.array(itr_true[['t'] + systems[sys_name].data_column_names])
    itr_true_inf = pd.read_csv(
        data_filepath + f"data_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv")
    itr_true_inf = np.array(itr_true_inf[['t'] + systems[sys_name].data_column_names])
    trajectories_true.append(itr_true)
    trajectories_true_inf.append(itr_true_inf)
    init_true = itr_true[0, 1:]
    time = itr_true[:, 0]

    # simulate

    sys_func = []
    for iexpr in expr_merged:
        if 'exp' in iexpr:
            iexpr = iexpr.replace('exp', 'np.exp')
        if 'log' in iexpr:
            iexpr = iexpr.replace('log', 'np.log')
        if 'sin' in iexpr:
            iexpr = iexpr.replace('sin', 'np.sin')
        if 'cos' in iexpr:
            iexpr = iexpr.replace('cos', 'np.cos')
        if 'Abs' in iexpr:
            iexpr = iexpr.replace('Abs', 'np.abs')
        if 'sqrt' in iexpr:
            iexpr = iexpr.replace('sqrt', 'np.sqrt')
        if '*tan' in iexpr:
            iexpr = iexpr.replace('*tan', '*np.tan')
        if ' tan' in iexpr:
            iexpr = iexpr.replace(' tan', ' np.tan')
        if '(tan' in iexpr:
            iexpr = iexpr.replace('(tan', '(np.tan')
        if '-tan' in iexpr:
            iexpr = iexpr.replace('-tan', '-np.tan')
        if '-arctan' in iexpr:
            iexpr = iexpr.replace('-arctan', '-np.arctan')
        if '*arctan' in iexpr:
            iexpr = iexpr.replace('*arctan', '*np.arctan')
        if ' arctan' in iexpr:
            iexpr = iexpr.replace(' arctan', ' np.arctan')
        if '(arctan' in iexpr:
            iexpr = iexpr.replace('(arctan', '(np.arctan')

        if sys_name == 'lorenz':
            sys_func.append(eval('lambda x_0, x_1, x_2: ' + iexpr))
        elif sys_name == 'cphase':
            sys_func.append(eval('lambda x_0, x_1, x_2: ' + iexpr))
        else:
            sys_func.append(eval('lambda x_0, x_1: ' + iexpr))

    def custom_func(t, x):
        return [sys_func[i](*x) for i in range(len(sys_func))]

    def custom_func_with_t(t, x):
        return [sys_func[i](t, *x) for i in range(len(sys_func))]

    np.random.seed(0)
    random.seed(0)

    if sys_name == 'cphase':
        simulation = odeint(custom_func_with_t, init_true, time, rtol=1e-12, atol=1e-12, tfirst=True)
    else:
        simulation = odeint(custom_func, init_true, time, rtol=1e-12, atol=1e-12, tfirst=True)

    trajectories_model.append(np.column_stack([time.reshape((len(time), 1)), simulation]))

    # trajectory error
    TEx = np.sqrt((np.mean((simulation[:, 0] - itr_true[:, 1]) ** 2))) / np.std(itr_true[:, 1])
    TEy = np.sqrt((np.mean((simulation[:, 1] - itr_true[:, 2]) ** 2))) / np.std(itr_true[:, 2])
    TExy = TEx + TEy
    if sys_name == 'lorenz':
        TEz = np.sqrt((np.mean((simulation[:, 2] - itr_true[:, 3]) ** 2))) / np.std(itr_true[:, 3])
        TExy = TExy + TEz

    # parameters
    true_params_unpacked = [j for i in systems[sys_name].gram_params for j in i]
    num_true_params = np.count_nonzero(true_params_unpacked)
    num_model_params = str(expr_merged).count('.')

    # join results in list
    results_transformer_clean_list.append(['transformer', 'sysident_num', data_version,
          sys_name, snr, iinit, 0, duration, TExy, num_true_params, num_model_params, expr_merged])

    # plot phase trajectories in a single figure and save
    if plot_trajectories_bool:
        plot_filepath = f"{path_base_out}plots{os.sep}"
        os.makedirs(plot_filepath, exist_ok=True)
        plot_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_{analy_version}_len{data_length}_" \
                        f"snr{snr}_init{iinit}_{{}}.{{}}"

        plot_trajectories(trajectories_true, trajectories_model, trajectories_true_inf,
                          fig_path=plot_filepath, fig_name=plot_filename, plot_one_example=True)

# put results in dataframe
column_names = ['method', 'exp_type', 'data_version', 'system', 'snr', 'iinit', 'ibest', 'duration',
                'TExy', 'num_true_params', 'num_model_params', 'expr_model']
results_transformer_clean = pd.DataFrame(results_transformer_clean_list, columns=column_names)

# save all results as a dataframe
results_transformer_clean.to_csv(f"{path_base_out}overall_results_table_{data_version}_"
                                 f"{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
content = tabulate(results_transformer_clean.values.tolist(), list(results_transformer_clean.columns), tablefmt="plain")
open(f"{path_base_out}overall_results_prettytable_{data_version}_{structure_version}_"
     f"{exp_version}_{analy_version}.csv", "w").write(content)
