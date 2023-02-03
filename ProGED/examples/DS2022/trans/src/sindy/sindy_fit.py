import os
import sys
import numpy as np
import pandas as pd
import time
import itertools
import pysindy as ps
from src.generate_data.systems_collection import strogatz, mysystems
from plotting.parestim_sim_plots import plot_trajectories
from tabulate import tabulate

np.random.seed(1)

def sindy_fit(X, t, x_dot=[], sys_name="default"):

    # optimizer
    #opt = ps.STLSQ(threshold=0.25)
    #opt = ps.SR3(threshold=0.10,
    #             max_iter=100000)

    if sys_name in ['bacres', 'glider', 'predprey', 'shearflow']:
        threshold = 0.10
    elif sys_name in ['vdpvdpUS', 'barmag']:
        threshold = 0.35
    else:
        threshold = 0.25

    opt = ps.SR3(
           threshold=threshold,
           thresholder="l1",
           max_iter=50000,
           normalize_columns=True,
           tol=1e-5
    )

    # library

    #  1. Initialize custom SINDy library up to sixth order polynomials

    # 1.1 library polynomial
    if sys_name in ['lv', 'vdp', 'myvdp', 'stl', 'lorenz', 'vdpvdpUS', 'stlvdpBL']:
        library_functions = [lambda x: x,
                             lambda x: x ** 2,
                             lambda x, y: x * y,
                             lambda x: x ** 3,
                             lambda x, y: x**2 * y,
                             lambda x, y: y**2 * x,
                             ]

        library_function_names = [lambda x: x,
                                  lambda x: x + '^2',
                                  lambda x, y: x + y,
                                  lambda x: x + '^3',
                                  lambda x, y: x + '^2' + y,
                                  lambda x, y: y + '^2' + x,
                                  ]
        if sys_name == 'lorenz':
            var_names = ["x", "y", "z"]
        elif sys_name in ['vdpvdpUS', 'stlvdpBL']:
            var_names = ["x", "y", "u", "v"]
        else:
            var_names = ["x", "y"]

    elif sys_name in ['bacres', 'glider', 'predprey']:

        library_functions = [lambda x: x,
                             lambda x: x ** 2,
                             lambda x: np.sin(x),
                             lambda x: np.cos(x),
                             lambda x, y: x * y,
                             lambda x, y: x / y,
                             lambda x, y: y / x,
                             lambda x, y: np.cos(y) / x,
                             lambda x: x ** 3,
                             lambda x, y: x**2 * y,
                             lambda x, y: y**2 * x
                             ]

        library_function_names = [lambda x: x,
                                  lambda x: x + '^2',
                                  lambda x: 'sin(' + x + ')',
                                  lambda x: 'cos(' + x + ')',
                                  lambda x, y: x + y,
                                  lambda x, y: x + '/' + y,
                                  lambda x, y: y + '/' + x,
                                  lambda x, y: 'cos(' + y + ')/(' + x + ')',
                                  lambda x: x + '^3',
                                  lambda x, y: x + '^2' + y,
                                  lambda x, y: y + '^2' + x
                                  ]

        var_names = ["x", "y"]
        # feature names: ['1', 'x', 'y', 'x^2', 'y^2', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'xy', 'x/y', 'y/x', 'cos(y)/(x)', 'x^3', 'y^3', 'x^2y', 'y^2x']

    elif sys_name in ['barmag', 'shearflow', 'cphase']:

        library_functions = [lambda x: np.sin(x),
                             lambda x: np.cos(x),
                             lambda x: np.tan(x),
                             lambda x: 1/np.tan(x),
                             lambda x,y: np.sin(x + y),
                             lambda x,y: np.sin(x - y),
                             lambda x,y: (np.sin(x)**2) * np.sin(y),
                             lambda x,y: (np.cos(x)**2) * np.sin(y),
                             lambda x,y: np.cos(x)*(1/np.tan(y)),
                             ]

        library_function_names = [lambda x: 'sin(' + x + ')',
                                  lambda x: 'cos(' + x + ')',
                                  lambda x: 'tan(' + x + ')',
                                  lambda x: 'cot(' + x + ')',
                                  lambda x, y: 'sin(' + x + '+' + y + ')',
                                  lambda x, y: 'sin(' + x + '-' + y + ')',
                                  lambda x, y: 'sin^2(' + y + ') * sin(' + x + ')',
                                  lambda x, y: 'cos^2(' + y + ') * sin(' + x + ')',
                                  lambda x, y: 'cos(' + x + ') cot(' + y + ')',
                                  ]
        if sys_name == 'cphase':
            var_names = ["x", "y", "t"]
        else:
            var_names = ["x", "y"]
            # ['1', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'sin(x+y)', 'sin(x-y)', '(sin(x)^2 + cos(x)^2)*sin(y)', 'cos(x) cot(y)']    else:
    else:
        print("Error. No library could be chosen based on the system name (sys_name). Recheck inputs.")

    lib = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
       # t=t,
        include_bias=True
    )

    mod = ps.SINDy(feature_names=var_names,
                   optimizer=opt,
                   feature_library=lib)
    if sys_name == 'cphase':
        mod.fit(X, t=t, u=t, x_dot=x_dot)  # , n_models=systems[sys_name].data_column_names)
    else:
        mod.fit(X, t=t, x_dot=x_dot)

    return mod


## MAIN

method="sindy"
exp_type="sysident_num"
systems = {**strogatz, **mysystems}
exp_version = "e2"
analy_version = 'a1'
structure_version = "s0"
data_version = "allonger"
set_obs = "full"  # either full, part or all
snrs = ["inf", 30, 20, 13, 10, 7]
inits = np.arange(0, 4)
data_length = 2000
sindy_threshold = 0.001
plot_trajectories_bool = True

sys_names = list(systems.keys())
combinations = []
for sys_name in sys_names:
    combinations.append(list(itertools.product([sys_name], systems[sys_name].get_obs(set_obs), inits, snrs)))
combinations = [item for sublist in combinations for item in sublist]

path_main = "D:\\Experiments\\MLJ23"
path_base_out = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
os.makedirs(path_base_out, exist_ok=True)

results_list = []

"""
sys_name = sys_names[7]
iobs = ['x', 'y']
snr = '10'
iinit = 0
"""

for sys_name, iobs, iinit, snr in combinations:
    benchmark = systems[sys_name].benchmark
    iobs_name = ''.join(iobs)
    print(f"{method} | {sys_name} | snr: {snr} | obs: {iobs_name} | init: {iinit}")

    # Get true clean data
    path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
    data_true_inf = pd.read_csv(path_data_in + f"data_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv")
    data_true_inf = np.array(data_true_inf[['t'] + systems[sys_name].data_column_names])
    # Get true noisy data
    data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
    data_orig = pd.read_csv(path_data_in + data_filename)
    data = np.array(data_orig[['t'] + systems[sys_name].data_column_names])
    col_names = list(data_orig.columns[:np.shape(data)[1]])
    # get derivatives
    data_der = np.array(data_orig[['d' + systems[sys_name].data_column_names[i] for i in range(len(systems[sys_name].data_column_names))]])

    X = data[:, 1:]
    t = data[:, 0]
    x0 = data[0, 1:]

    startTime = time.time()

    try:
        # run sindy
        model = sindy_fit(X,  t, x_dot=data_der, sys_name=sys_name)
        model.print()
    except:
        ires = [method, data_version, exp_version, benchmark, sys_name, snr, iobs_name, iinit] + \
               list(np.full(16, np.nan))
        results_list.append(ires)
        print('Error in sindy_fit. Continue with next system')
        continue

    # check duration
    endTime = time.time()
    duration = endTime - startTime

    # simulate
    try:
        sim = model.simulate(x0, t, integrator="odeint")
    except:
        ires = [method, data_version, exp_version, benchmark, sys_name, snr, iobs_name, iinit] + \
               list(np.full(16, np.nan))
        results_list.append(ires)
        print('Error in model.simulate. Continue with next system')
        continue

    if len(sim) != len(t) or np.any(np.isnan(sim)):
        print(sys_name, snr, iinit, "Simulation failed due to size mismatch or nans in simulation.")
        ires = [method, data_version, exp_version, benchmark, sys_name, snr, iobs_name, iinit] + \
               list(np.full(16, np.nan))
        results_list.append(ires)
        continue

    # save simulation
    path_out = path_base_out + sys_name + os.sep
    os.makedirs(path_out, exist_ok=True)
    out_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}" \
                   f"_snr{snr}_init{iinit}_obs{iobs_name}_{{}}.{{}}"
    sim_with_t = np.hstack((t.reshape(t.size, 1), sim))
    simulation = pd.DataFrame(sim_with_t, columns=col_names)
    simulation.to_csv(path_out + out_filename.format(f"simulation_{iinit}", "csv"), index=None)

    # trajectory error
    TEx = np.sqrt((np.mean((sim_with_t[:, 1] - data[:, 1]) ** 2))) / np.std(data[:, 1])
    TEy = np.sqrt((np.mean((sim_with_t[:, 2] - data[:, 2]) ** 2))) / np.std(data[:, 2])
    TExy = TEx + TEy

    # reconstruction error
    modeled_params_unpacked = [j for i in model.coefficients() for j in i]
    RE = np.sqrt(np.mean(np.subtract(modeled_params_unpacked, systems[sys_name].sindy_params) ** 2))

    #
    num_true_params = len(np.nonzero(systems[sys_name].sindy_params)[0])
    num_model_params = np.sum(np.array(modeled_params_unpacked) > sindy_threshold)

    ires = [method, data_version, exp_version, benchmark, sys_name, snr, iobs_name, iinit, x0, x0,
            duration, np.nan, np.nan, np.nan, systems[sys_name].orig_expr, [model.equations()],
            systems[sys_name].sym_params, modeled_params_unpacked,
            TEx, TEy, TExy, RE, num_true_params, num_model_params]

    results_list.append(ires)

    # plot phase trajectories in a single figure and save
    if plot_trajectories_bool:
        plot_filepath = f"{path_out}plots{os.sep}"
        plot_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}_snr{snr}_init{iinit}_obs{iobs_name}_{{}}.{{}}"
        os.makedirs(plot_filepath, exist_ok=True)
        plot_trajectories([data], [sim_with_t], [data_true_inf], fig_path=plot_filepath, fig_name=plot_filename)


# list -> dataframe, all results
results = pd.DataFrame(results_list,
    columns=["method", "data_version", "exp_version", "benchmark", "system", "snr", "obs_type", "iinit", "init_true", "init_model",
             "duration", "rmseDE", "max_iter", "pop_size", "expr_true", "expr_model", "params_true", "params_model", "TEx", "TEy", "TExy", "RE",
             "num_true_params", "num_model_params"])

# save all results as a dataframe
results.to_csv(f"{path_base_out}overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
content = tabulate(results.values.tolist(), list(results.columns), tablefmt="plain")
open(f"{path_base_out}overall_results_prettytable_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", "w").write(content)

print("finished")