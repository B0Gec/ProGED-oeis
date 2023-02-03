import sys
import os
import pickle as pkl
import pandas as pd
import numpy as np
import ProGED as pg
import itertools
from src.generate_data.systems_collection import strogatz, mysystems
from proged.helper_functions import get_fit_settings

np.random.seed(1)

def get_settings(iinput, systems, snrs, inits, set_obs):
    sys_names = list(systems.keys())

    combinations = []
    for sys_name in sys_names:
        obss=systems[sys_name].get_obs(set_obs)
        batches = np.arange(systems[sys_name].num_batches)
        combinations.append(list(itertools.product([sys_name], obss, inits, snrs, batches)))
    combinations = [item for sublist in combinations for item in sublist]
    return combinations[iinput-1]

# metod info
method = 'proged'
type = 'sysident_num'

# data info
systems = {**strogatz, **mysystems}
exp_version = "e1"
data_version = "all"
set_obs = "full"  # either full, part or all
snrs = ["inf"] #, 30, 20, 13, 10, 7]
inits = np.arange(0, 4)
data_length = 100

# structures info
struct_version = "s4"

# decide on the job to do
iinput = 2 # int(sys.argv[1])
sys_name, iobs, iinit, snr, ib = get_settings(iinput, systems, snrs, inits, set_obs)
iobs_name = ''.join(iobs)

path_main = "D:\\Experiments\\MLJ23"
path_base = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{type}{os.sep}"
path_results = f"{path_base}{exp_version}{os.sep}"

# Get data
path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
data_orig = pd.read_csv(path_data_in + data_filename)
data_header = list(data_orig.columns)

# Fit each equation separately
# ieq, eqsym = 0, systems[sys_name].data_column_names[0]
for ieq, eqsym in enumerate(systems[sys_name].data_column_names):
    print(f"{sys_name} | snr: {snr} | init: {iinit} | batch: {ib} | eq: {eqsym}")

    # find the right columns in data
    data_idx = [data_header.index(systems[sys_name].sys_vars[i]) for i in range(len(systems[sys_name].sys_vars))]
    target_idx = data_header.index('d' + eqsym)
    indices = np.unique(data_idx + [target_idx])
    target_idx_new = np.where(target_idx == indices)[0][0]
    data_eq = np.array(data_orig)[:, indices]

    # get info on structures
    grammars = systems[sys_name].grammar_type
    if len(grammars) > 1:
        gram = grammars[0] if eqsym in str(systems[sys_name].grammar_vars[0][0][:2]) else grammars[1]
    else:
        gram = grammars[0]

    num_samples = systems[sys_name].num_samples
    num_batches = systems[sys_name].num_batches
    path_structures = f"{path_base}structures{os.sep}{struct_version}{os.sep}" \
                      f"structs_{struct_version}_{gram}_nsamp{num_samples}_nbatch{num_batches}{os.sep}"
    systemBox = pg.ModelBox(observed=iobs)
    systemBox.load(f"{path_structures}structs_{struct_version}_{gram}_nsamp{num_samples}_nbatch{num_batches}_b{ib}.pg")

    # settings for parameter estimation
    estimation_settings = get_fit_settings(obs=iobs)
    estimation_settings["target_variable_index"] = target_idx_new
    estimation_settings["optimizer_settings"]["lower_upper_bounds"] = systems[sys_name].bounds

    # parameter estimation
    systemBox_fitted = pg.fit_models(systemBox, data_eq, task_type='algebraic', estimation_settings=estimation_settings)
    systemBox_fitted.observed = iobs

    # save the fitted models and the settings file
    path_out = f"{path_results}{sys_name}{os.sep}"
    os.makedirs(path_out, exist_ok=True)
    out_filename = f"{sys_name}_{data_version}_{struct_version}_{exp_version}_len{data_length}" \
                   f"_snr{snr}_init{iinit}_obs{iobs_name}_b{ib}_{eqsym}_fitted.pg"
    systemBox_fitted.dump(path_out + out_filename)

# save settings of this exp
if iinput == 1:

    settings_filename = f"estimation_settings_{data_version}_{struct_version}_{exp_version}"

    with open(path_results + settings_filename + ".pkl", "wb") as set_file:
        pkl.dump(estimation_settings, set_file)

    fo = open(path_results + settings_filename + ".txt", "w")
    for k, v in estimation_settings.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    fo.close()

    systems_filename = f"systems_settings_{data_version}_{struct_version}_{exp_version}.txt"
    fo = open(path_results + systems_filename, "w")
    for k, v in systems.items():
        fo.write(str(k) + ' >>> ' + str(systems[str(k)].__dict__) + '\n\n')
    fo.close()

print(" -- Finished -- ")

