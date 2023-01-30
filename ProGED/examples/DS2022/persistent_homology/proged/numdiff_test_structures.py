import sys
import os
import time
import pickle
import pandas as pd
import numpy as np
import ProGED as pg
from src.generate_data import systems_collection
from proged.MLJ_helper_functions import get_fit_settings

data_version = "strogatz"
structure_version = "0v"
exp_version = "v1"
path_main = "D:\\Experiments\\MLJ23\\proged"
# path_main = "./"
path_base_out = f"{path_main}\\numdiff\\identification\\{exp_version}"

grammars = ["polynomial", "phaseosc"]
types = ["onestate", "lorenz", "twophase", "twostate1", "twostate2"]
sys_names = ["vdp", "stl", "lotka", "cphase", "lorenz", "vdpvdpUS", "stlvdpBL"]
num_samples = 10000
n_batches = 10
# batch_idx = '1'
snrs = ['inf', 30, 20, 13, 10, 7]          # set noise level
set_obs = "full"
data_length = 60
n_inits = 1

# ----- Get the settings for a particular system fitting job
sys_name, isnr, iinit, ibatch, sys_true_expr, sys_true_params, sys_bounds, sys_symbols, \
    iobs, iobstxt, igram, igramtype = set_numdiff_input(snrs, sys_names, n_inits, n_batches, iinput)

num_eq = len(sys_symbols)-1 if "t" in sys_symbols else len(sys_symbols)

# ----- Get data -------
path_data_in = f"{path_main}\\data\\{data_version}\\{sys_name}\\"
data_filename = f"{sys_name}_dat{data_version}_len{data_length}_snr{isnr}_init{iinit}_data_withder.csv"
data = pd.read_csv(path_data_in + data_filename)
data_header = list(data.columns)

# ------- Import models in a specific batch made by specific grammar --------
# itype = igramtype[1]
for itype in igramtype:
    path_batch_in = f"{path_main}\\numdiff\\structures\\{structure_version}\\batchsize{n_batches}\\{itype}\\"
    batch_filename = "structures_{}_{}_{}_n{}_batch{}.pg".format(structure_version, igram, itype, num_samples, ibatch)
    with open(os.path.join(path_batch_in, batch_filename), "rb") as file:
        models = pickle.load(file)

    # --------  Fit each equation in a system separately ---------
    # eq_idx = 1
    # eq_sym = sys_symbols[eq_idx]
    for eq_idx, eq_sym in enumerate(sys_symbols[:num_eq]):

        # ------- Estimate parameters --------
        print("--Fitting models")
        print(f"{sys_name} | snr: {isnr} | init: {iinit} | batch: {ibatch} | type: {itype} | eq: {eq_sym}")

        # find the right columns in data
        if itype == "lorenz":
            X_idx = [1, 2, 3]
        elif "twostate" in itype:
            X_idx = [1, 2, 3, 4]
        else:
            X_idx = [1, 2]
        [X_idx.append(col_idx) for col_idx, col_name in enumerate(data_header) if 'd' + eq_sym in col_name]

        # get target idx (derivative (the index found above) is the target)
        target_idx = len(X_idx) - 1

        # make sure the columns of data (X) follow the order of system symbols (specially if time is included)
        if "t" in sys_symbols:
            X_idx.append(0)

        # set data for one equation (only target derivative included)
        data_eq = np.array(data)[:, X_idx]

        # get estimation settings
        estimation_settings = get_fit_settings()
        estimation_settings["target_variable_index"] = target_idx
        estimation_settings["optimizer_settings"]["lower_upper_bounds"] = sys_bounds

        # fit
        start_time = time.time()
        models_out = pg.fit_models(models,
                                   data = data_eq,
                                   task_type='algebraic',
                                   estimation_settings=estimation_settings)
        print("--End time in seconds: " + str(time.time() - start_time))

        # save the fitted models and the settings file
        path_out = f"{path_base_out}\\{sys_name}\\"
        os.makedirs(path_out, exist_ok=True)
        out_filename = f"{sys_name}_dat{data_version}_str{structure_version}_exp{exp_version}_len{data_length}_" \
                       f"{iobstxt}_{itype}_snr{isnr}_init{iinit}_ib{ibatch}_eq{eq_sym}_fitted.pg"
        models_out.dump(path_out + out_filename)

# save settings of this exp
if int(iinput) == 1:
    settings_filename = f"\\estimation_settings_dat{data_version}_str{structure_version}_exp{exp_version}.txt"
    fo = open(path_base_out + settings_filename, "w")
    for k, v in estimation_settings.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    fo.close()


