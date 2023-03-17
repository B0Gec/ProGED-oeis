import os
import numpy
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.integrate import odeint
import ProGED as pg
from src.generate_data.systems_collection import strogatz, mysystems
from src.generate_data.system_obj import System
from plotting.parestim_sim_plots import plot_trajectories, plot_optimization_curves

##
method = "proged"
exp_type = 'sysident_num'

systems = {**strogatz, **mysystems}
data_version = 'all'
exp_version = 'e2'
structure_version = 's7'
analy_version = 'a1'
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"

sys_names = list(systems.keys())
snrs = ['inf', 30, 13]
set_obs = "full"
n_inits = 4
sim_step = 0.1
sim_time = 10
data_length = int(sim_time / sim_step)
plot_trajectories_bool = True
plot_optimization_curves_bool = False

results_list = []

# sys_name, snr, iobs, obs, iinit, im, ieq, eqsym, ib = sys_names[6], snrs[0], 0, ['x', 'y'], 0, 0, 0, 'x', 0
for sys_name in sys_names:
    sys_name = "cphase"
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
                    itr_true = np.array(itr_true[['t'] + systems[sys_name].data_column_names])
                    itr_true_inf = pd.read_csv(
                        data_filepath + f"data_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv")
                    itr_true_inf = np.array(itr_true_inf[['t'] + systems[sys_name].data_column_names])
                    init_true = itr_true[0, 1:]
                except:
                    print("Not Valid Data File: " + data_filename)
                    itr_true, init_true = None, None

                trajectories_true.append(itr_true)
                trajectories_true_inf.append(itr_true_inf)

                # get estimation settings
                estimation_settings = pd.read_pickle(settings_filename)

                best_models = []
                for ieq, eqsym in enumerate(systems[sys_name].data_column_names):
                    print(f"Started: {sys_name} | snr: {snr} | iobs: {obs_name} | init: {iinit} | eq: {eqsym}")

                    models = pg.ModelBox()
                    for ib in range(systems[sys_name].num_batches):
                        file_name = f"{sys_name}_{data_version}_" \
                                    f"{structure_version}_{exp_version}_len{data_length}_snr{snr}_" \
                                    f"init{iinit}_obs{obs_name}_b{ib}_{eqsym}_fitted.pg"
                        try:
                            models.load(f"{path_base_in}{sys_name}{os.sep}{file_name}")
                        except:
                            print(f"not loaded correctly: {file_name}")

                    # GET BEST MODELS
                    N = 100
                    ibest_models = models.retrieve_best_models(N=N)
                    best_models.append(ibest_models)

                    best_models_filename = f"best_models_{sys_name}_{data_version}_" \
                                           f"{structure_version}_{exp_version}_len{data_length}_snr{snr}_" \
                                           f"obs{obs_name}_{eqsym}.txt"
                    with open(f"{path_base_in}{sys_name}{os.sep}{best_models_filename}", "a") as f:
                        f.write("\n---" + best_models_filename + ", unique models: " + str(len(best_models)) + "\n")
                        f.write("\n---\n--- Original ---")
                        f.write("\n" + str(systems[sys_name].gram_structure))
                        f.write("\n" + str(systems[sys_name].orig_expr))
                        f.write("\n---\n---looking at RMSE DE ---")
                        f.write(str(ibest_models))

                    del models

                # generate best models, every eq with every eq:
                # i, iexpr, j, jexpr = 0, best_models[0][0], 0, best_models[0][0]

                evaluated_models = []
                evaluation = []
                for i, iexpr in enumerate(best_models[0]):
                    for j, jexpr in enumerate(best_models[1]):
                        print([i,j])
                        ijmodel=System(sys_name, best_models[0][i].sym_vars)
                        sys_func = [iexpr.lambdify(), jexpr.lambdify()]
                        t = np.arange(0, sim_time, sim_step)

                        def custom_func(t, x):
                            return [sys_func[i](*x) for i in range(len(sys_func))]

                        simulation = odeint(custom_func, init_true, t, rtol=1e-12, atol=1e-12, tfirst=True)

                        TEx = np.sqrt((np.mean((simulation[:, 0] - itr_true[:, 1]) ** 2))) / np.std(itr_true[:, 1])
                        TEy = np.sqrt((np.mean((simulation[:, 1] - itr_true[:, 2]) ** 2))) / np.std(itr_true[:, 2])
                        ijmodel.TExy = TEx + TEy
                        ijmodel.expr_merged = [iexpr, jexpr]
                        ijmodel.params_merged = [iexpr.params, jexpr.params]
                        ijmodel.duration_merged = sum([iexpr.estimated['duration'], jexpr.estimated['duration']])
                        ijmodel.obj_fun_error_merged = [iexpr.estimated['fun'], jexpr.estimated['fun']]
                        ijmodel.index = [i, j]
                        evaluated_models.append(ijmodel)
                        evaluation.append(ijmodel.TExy)

                # sort the models
                evaluation = np.array(evaluation)
                best_systems_idx = np.argsort(evaluation)[:10]

                # ibest = best_systems_idx[0]
                for ibest in best_systems_idx:
                    sys_func = [evaluated_models[ibest].expr_merged[0].lambdify(), evaluated_models[ibest].expr_merged[1].lambdify()]
                    t = np.arange(0, sim_time, sim_step)

                    def custom_func(t, x):
                        return [sys_func[i](*x) for i in range(len(sys_func))]

                    itr_model = odeint(custom_func, init_true, t, rtol=1e-12, atol=1e-12, tfirst=True)
                    trajectories_model.append(np.column_stack([t.reshape((len(t), 1)), itr_model]))

                    # get errors - trajectory error
                    TEx = np.sqrt((np.mean((itr_model[:, 0] - itr_true[:, 1]) ** 2))) / np.std(itr_true[:, 1])
                    TEy = np.sqrt((np.mean((itr_model[:, 1] - itr_true[:, 2]) ** 2))) / np.std(itr_true[:, 2])
                    TExy = TEx + TEy

                    # get errors - parameter reconstruction error
                    modeled_params_unpacked = [j for i in evaluated_models[ibest].params_merged for j in i]
                    true_params_unpacked = [j for i in systems[sys_name].gram_params for j in i]
                    RE = np.nan  # np.sqrt(np.mean(np.subtract(modeled_params_unpacked, true_params_unpacked) ** 2))

                    num_true_params = np.count_nonzero(true_params_unpacked)
                    num_model_params = np.count_nonzero(modeled_params_unpacked)

                    # gather all results
                    ires = [method, data_version, exp_version, benchmark, sys_name, snr, obs_name, iinit, ibest, init_true,
                            init_true,
                            evaluated_models[ibest].duration_merged, evaluated_models[ibest].obj_fun_error_merged, estimation_settings["optimizer_settings"]["max_iter"],
                            estimation_settings["optimizer_settings"]["pop_size"], systems[sys_name].orig_expr, evaluated_models[ibest].expr_merged,
                            systems[sys_name].sym_params, evaluated_models[ibest].params_merged, TEx, TEy, TExy, RE, num_true_params,
                            num_model_params]

                    results_list.append(ires)

                    # plot phase trajectories in a single figure and save
                    plot_filepath = f"{results_filepath}{sys_name}{os.sep}plots{os.sep}"
                    plot_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}_snr{snr}_init{iinit}_obs{obs_name}_{{}}.{{}}"
                    os.makedirs(plot_filepath, exist_ok=True)

                    if plot_trajectories_bool:
                        plot_trajectories(trajectories_true, trajectories_model, trajectories_true_inf,
                                          fig_path=plot_filepath, fig_name=plot_filename, plot_one_example=True)

                del evaluated_models, evaluation, best_models
                # if (iinit == 0 or iinit == 3) and plot_optimization_curves_bool and systemBox[0].valid:
                #    plot_optimization_curves(systemBox, fig_path=plot_filepath, fig_name=plot_filename, optimizer=estimation_settings["optimizer"])

    # list -> dataframe, all results
    results = pd.DataFrame(results_list,
                           columns=["method", "data_version", "exp_version", "benchmark", "system", "snr", "obs_type",
                                    "iinit", "ibest", "init_true", "init_model",
                                    "duration", "rmseDE", "max_iter", "pop_size", "expr_true", "expr_model", "params_true",
                                    "params_model", "TEx", "TEy", "TExy", "RE",
                                    "num_true_params", "num_model_params"])

    # save all results as a dataframe
    results.to_csv(f"{results_filepath}overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}_{sys_name}_with_best100.csv", sep='\t')
    content = tabulate(results.values.tolist(), list(results.columns), tablefmt="plain")
    open(f"{results_filepath}overall_results_prettytable_{data_version}_{structure_version}_{exp_version}_{analy_version}_{sys_name}_with_best100.csv","w").write(content)

    print("finished")
