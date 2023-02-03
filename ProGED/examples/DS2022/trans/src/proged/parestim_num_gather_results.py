# Code to gather results of the (only) parameter estimation in the table.

import os

import numpy
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tabulate import tabulate
import ProGED as pg
from src.generate_data.systems_collection import strogatz, mysystems
from src.generate_data.system_obj import System
from plotting.parestim_sim_plots import plot_trajectories, plot_optimization_curves

##
method = "proged"
exp_type = 'parestim_num'

systems = {**strogatz, **mysystems}
data_version = 'all'
exp_version = 'e1'
structure_version = 's0'
analy_version = 'a1'
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"

sys_names = list(systems.keys())
snrs = ['inf', 30, 20, 13, 10, 7]
set_obs = "full"
n_inits = 4
sim_step = 0.1
sim_time = 10
data_length = int(sim_time/sim_step)
plot_trajectories_bool = True
plot_optimization_curves_bool = False

results_list = []

# sys_name, snr, iobs, obs, iinit, im = sys_names[0], snrs[0], 0, ['x', 'y'], 0, 0
for sys_name in sys_names:
    obss = systems[sys_name].get_obs(set_obs)
    benchmark = systems[sys_name].benchmark
    for snr in snrs:
        for iobs, obs in enumerate(obss):
            obs_name = "".join(obs)

            for iinit in range(n_inits):

                trajectories_true = []
                trajectories_model = []
                trajectories_true_inf = []

                # set some naming / paths
                data_filepath = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
                data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
                results_filepath = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
                results_filename = f"{sys_name}{os.sep}{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}_snr{snr}_init{iinit}_obs{obs_name}_fitted.pg"
                settings_filename = f"{results_filepath}estimation_settings_{data_version}_{structure_version}_{exp_version}.pkl"

                # get true data (gather true trajectories) & get true initial values
                try:
                    itr_true = pd.read_csv(data_filepath + data_filename)
                    itr_true =  np.array(itr_true[['t'] + systems[sys_name].data_column_names])
                    itr_true_inf = pd.read_csv(data_filepath + f"data_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv")
                    itr_true_inf = np.array(itr_true_inf[['t'] + systems[sys_name].data_column_names])
                    init_true = itr_true[0, 1:]
                except:
                    print("Not Valid Data File: " + data_filename)
                    itr_true, init_true = None, None

                trajectories_true.append(itr_true)
                trajectories_true_inf.append(itr_true_inf)

                # get estimation settings
                estimation_settings = pd.read_pickle(settings_filename)

                # get results
                try:
                    systemBox = pg.ModelBox(observed=obs)
                    systemBox.load(results_filepath + results_filename)
                    print("Loaded: " + results_filename)
                except:
                    print("Skipping: " + results_filename)
                    ires = [method, data_version, exp_version,
                            benchmark, sys_name, snr, obs_name, iinit] + \
                            list(np.full(len(results_list[0]) - 7, np.nan))
                    results_list.append(ires)
                    continue

                # go over all estimated valid models in a model box
                expr_merged = [systemBox[im].expr for im in range(len(systemBox))]
                params_merged = [systemBox[im].params for im in range(len(systemBox))]
                if sys_name == 'vdpvdpUS':
                    params_merged[1] = np.append(params_merged[1], 1.001)
                    params_merged[3] = np.insert(params_merged[3], 2, 1.001)
                duration_merged = 0
                for im in range(len(systemBox)):
                    if type(systemBox[im].estimated['fun']) == np.float64:
                        systemBox[im].estimated['fun'] = [systemBox[im].estimated['fun']]
                    if 'duration' in list(systemBox[im].estimated.keys()):
                        duration_merged += systemBox[im].estimated['duration']

                obj_fun_error_merged = np.mean([systemBox[im].estimated['fun'] for im in range(len(systemBox))])
                systemBox_merged = pg.ModelBox(observed=obs)
                systemBox_merged.add_system(expr_merged,
                                            symbols={"x": systems[sys_name].sys_vars, "const": "C"},
                                            params=params_merged)
                systemBox_merged[0].valid=True

                # gather modeled trajectories
                sys_func = systemBox_merged[0].lambdify(list=True)
                system_model = System(sys_name="model",
                                      sys_vars=systemBox[0].sym_vars,
                                      sys_func=sys_func)
                itr_model = system_model.simulate(init_true, sim_step=sim_step, sim_time=sim_time)
                trajectories_model.append(itr_model)

                # get errors - trajectory error
                TEx = np.sqrt((np.mean((itr_model[:, 1] - itr_true[:, 1]) ** 2))) / np.std(itr_true[:, 1])
                TEy = np.sqrt((np.mean((itr_model[:, 2] - itr_true[:, 2]) ** 2))) / np.std(itr_true[:, 2])
                TExy = TEx + TEy

                # get errors - parameter reconstruction error
                modeled_params_unpacked = [j for i in params_merged for j in i]
                true_params_unpacked = [j for i in systems[sys_name].sym_params for j in i]
                RE = np.sqrt(np.mean(np.subtract(modeled_params_unpacked, true_params_unpacked) ** 2))

                num_true_params = np.count_nonzero(true_params_unpacked)
                num_model_params = np.count_nonzero(modeled_params_unpacked)

                # gather all results
                ires = [method, data_version, exp_version, benchmark, sys_name, snr, obs_name, iinit, init_true, init_true,
                        duration_merged, obj_fun_error_merged, estimation_settings["optimizer_settings"]["max_iter"],
                        estimation_settings["optimizer_settings"]["pop_size"], systems[sys_name].orig_expr, expr_merged,
                        systems[sys_name].sym_params, params_merged, TEx, TEy, TExy, RE, num_true_params, num_model_params]

                results_list.append(ires)

            # plot phase trajectories in a single figure and save
            plot_filepath = f"{results_filepath}{sys_name}{os.sep}plots{os.sep}"
            plot_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}_snr{snr}_init{iinit}_obs{obs_name}_{{}}.{{}}"
            os.makedirs(plot_filepath, exist_ok=True)

            if (iinit == 0 or iinit == 3) and plot_trajectories_bool and systemBox[0].valid:
                plot_trajectories(trajectories_true, trajectories_model, trajectories_true_inf, fig_path=plot_filepath, fig_name=plot_filename)

            if (iinit == 0 or iinit == 3) and plot_optimization_curves_bool and systemBox[0].valid:
                plot_optimization_curves(systemBox, fig_path=plot_filepath, fig_name=plot_filename, optimizer=estimation_settings["optimizer"])

# list -> dataframe, all results
results = pd.DataFrame(results_list,
    columns=["method", "data_version", "exp_version", "benchmark", "system", "snr", "obs_type", "iinit", "init_true", "init_model",
             "duration", "rmseDE", "max_iter", "pop_size", "expr_true", "expr_model", "params_true", "params_model", "TEx", "TEy", "TExy", "RE",
             "num_true_params", "num_model_params", "b"])

# save all results as a dataframe
results.to_csv(f"{results_filepath}overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
content = tabulate(results.values.tolist(), list(results.columns), tablefmt="plain")
open(f"{results_filepath}overall_results_prettytable_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", "w").write(content)

print("finished")