import sys
import os
import pickle as pkl
import pandas as pd
import numpy as np
import ProGED as pg
import itertools
from generate_data.systems_collection import strogatz, mysystems
from MLJ_helper_functions import get_fit_settings

np.random.seed(1)

def get_settings(iinput, systems, snrs, inits, set_obs):
    sys_names = list(systems.keys())
    combinations = []
    for sys_name in sys_names:
        combinations.append(list(itertools.product([sys_name], systems[sys_name].get_obs(set_obs), inits, snrs)))
    combinations = [item for sublist in combinations for item in sublist]
    return combinations[iinput-1]

type = 'parestim_num'
method = 'proged'
iinput = int(sys.argv[1])
#iinput = 1
systems = {**strogatz, **mysystems}
exp_version = "e1"
data_version = "all"
structure_version = "s0"
set_obs = "full"  # either full, part or all
snrs = ["inf", 30, 20, 13, 10, 7]
inits = np.arange(0, 4)
sys_name, iobs, iinit, snr = get_settings(iinput, systems, snrs, inits, set_obs)
iobs_name = ''.join(iobs)

if data_version == "all":
    data_length = 100
elif data_version == "allong":
    data_length = 1000
elif data_version == "allonger":
    data_length = 2000

path_main = ""
path_base_out = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{type}{os.sep}{exp_version}{os.sep}"

# Get data
path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
data_orig = pd.read_csv(path_data_in + data_filename)
data_header = list(data_orig.columns)

# fit each equation separately
systemBoxes = pg.ModelBox(observed=iobs)

# ieq, eqsym = 0, systems[sys_name].data_column_names[0]
for ieq, eqsym in enumerate(systems[sys_name].data_column_names):

    print(f"{sys_name} | snr: {snr} | init: {iinit} | eq: {eqsym}")

    # find the right columns in data
    data_idx = [data_header.index(systems[sys_name].sys_vars[i]) for i in range(len(systems[sys_name].sys_vars))]
    target_idx = data_header.index('d' + eqsym)
    indices = np.unique(data_idx + [target_idx])
    target_idx_new = np.where(target_idx == indices)[0][0]
    data_eq = np.array(data_orig)[:, indices]

    estimation_settings = get_fit_settings(obs=iobs)
    estimation_settings["target_variable_index"] = target_idx_new
    estimation_settings["optimizer_settings"]["lower_upper_bounds"] = systems[sys_name].bounds

    # put system in proged env
    systemBox = pg.ModelBox(observed=iobs)
    systemBox.add_model(systems[sys_name].sym_structure[ieq], symbols={"x": systems[sys_name].sys_vars, "const": "C"})
    systemBox_fitted = pg.fit_models(systemBox, data_eq, task_type='algebraic', estimation_settings=estimation_settings)
    systemBox_fitted.observed = iobs

    # append to the overall output files
    systemBoxes.models_dict[ieq] = systemBox_fitted[0]

# save the fitted models and the settings file
path_out = f"{path_base_out}{sys_name}{os.sep}"
os.makedirs(path_out, exist_ok=True)
out_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}" \
               f"_snr{snr}_init{iinit}_obs{iobs_name}_fitted.pg"
systemBoxes.dump(path_out + out_filename)

print('finished')

# save settings of this exp
if iinput == 1:

    settings_filename = f"estimation_settings_{data_version}_{structure_version}_{exp_version}"

    with open(path_base_out + settings_filename + ".pkl", "wb") as set_file:
        pkl.dump(estimation_settings, set_file)

    fo = open(path_base_out + settings_filename + ".txt", "w")
    for k, v in estimation_settings.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    fo.close()

    systems_filename = f"systems_settings_{data_version}_{structure_version}_{exp_version}.txt"
    fo = open(path_base_out + systems_filename, "w")
    for k, v in systems.items():
        fo.write(str(k) + ' >>> ' + str(systems[str(k)].__dict__) + '\n\n')
    fo.close()

