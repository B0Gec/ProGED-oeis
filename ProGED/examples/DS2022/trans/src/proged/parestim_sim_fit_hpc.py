import sys
import os
import pickle as pkl
import pandas as pd
import numpy as np
import ProGED as pg
import itertools
from generate_data.systems_collection import strogatz, mysystems
from MLJ_helper_functions import get_fit_settings

def get_settings(iinput, systems, snrs, inits, set_obs):
    sys_names = list(systems.keys())
    combinations = []
    for sys_name in sys_names:
        combinations.append(list(itertools.product([sys_name], systems[sys_name].get_obs(set_obs), inits, snrs)))
    combinations = [item for sublist in combinations for item in sublist]
    return combinations[iinput-1]


iinput = int(sys.argv[1])
systems = {**strogatz, **mysystems}
data_version = "allong"
exp_version = "e2"
structure_version = "s0"

set_obs = "all"  # either full, part or all
snrs = ["inf", 30, 20, 13, 10, 7]
inits = np.arange(0, 4)
sys_name, iobs, iinit, snr = get_settings(iinput, systems, snrs, inits, set_obs)

if data_version == "all":
    data_length = 100
elif data_version == "allong":
    data_length = 1000
elif data_version == "allonger":
    data_length = 2000
    
path_main = "."
path_base_out = f"{path_main}{os.sep}results{os.sep}proged{os.sep}parestim_sim{os.sep}{exp_version}{os.sep}"

# ----- Get data (without der) -------
path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
data_orig = pd.read_csv(path_data_in + data_filename)
 
# prepare data
iobs_name = ''.join(iobs)
data = np.array(pd.concat([data_orig.iloc[:, 0], data_orig[iobs]], axis=1))

# fit
print(f"{iinput}_{sys_name} | snr: {snr} | obs: {iobs_name} | init: {iinit}")
estimation_settings = get_fit_settings(obs=iobs)
estimation_settings["optimizer_settings"]["lower_upper_bounds"] = systems[sys_name].bounds

systemBox = pg.ModelBox(observed=iobs)
systemBox.add_system(systems[sys_name].sym_structure, symbols={"x": systems[sys_name].sys_vars, "const": "C"})
systemBox_fitted = pg.fit_models(systemBox, data, task_type='differential', estimation_settings=estimation_settings)
systemBox_fitted.observed = iobs

# save the fitted models and the settings file
path_out = f"{path_base_out}{sys_name}{os.sep}"
os.makedirs(path_out, exist_ok=True)
out_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}" \
               f"_snr{snr}_init{iinit}_obs{iobs_name}_fitted.pg"
systemBox_fitted.dump(path_out + out_filename)

# save settings of this exp (just once)
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
