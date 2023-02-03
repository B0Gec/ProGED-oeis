# Generate data of different dynamical systems for experiments

import os
import numpy as np
import pandas as pd
import ProGED as pg
from src.generate_data.add_noise_to_data import add_noise_to_data, plot_noisy_data
from src.generate_data.systems_collection import strogatz, mysystems
from extensisq import BS45, BS45_i

systems = {**strogatz, **mysystems}
np.random.seed(1)
data_version = "all_test"
path_base_out = f"D:\\Experiments\\MLJ23\\data\\{data_version}"
sys_names = list(systems.keys())
snrs = ['inf'] #, 30, 20, 13, 10, 7]          # set signal to noise ratio (60 the best, 0 the worse) [inf, 60, 50, 40, 30, 20, 10, 0]

sim_time = 10
sim_step = 0.01
data_length = int(sim_time/sim_step)
calculate_derivatives = True
n_inits = 4
inits_manual = True
manual_inits_path_in = f"D:\\Experiments\\MLJ23\\data\\all\\"
make_plots = True

# sys_name, snr, iinit = sys_names[2], snrs[0], 0
for sys_name in sys_names:
    for snr in snrs:
        inits_range = n_inits if n_inits != 0 else len(systems[sys_name].inits)
        for iinit in range(inits_range):

            if inits_manual:
                #data_true = pd.read_csv(true_data_path_in + sys_name + "\\" + sys_name + str(iinit) + ".csv")
                data_true = pd.read_csv(manual_inits_path_in + sys_name + "\\" + "data_" + sys_name + "_all_len100_snrinf_init" + str(iinit) + ".csv")
                systems[sys_name].inits = np.array(data_true[systems[sys_name].data_column_names])[0]
                init=systems[sys_name].inits
            else:
                init = iinit

            # simulate
            data = systems[sys_name].simulate(init, sim_step=sim_step, sim_time=sim_time)

            # add noise
            if snr != 'inf':
                data = add_noise_to_data(data, target_snr_db=snr)

            # save data
            path_out = f"{path_base_out}\\{sys_name}\\"
            os.makedirs(path_out, exist_ok=True)
            data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}"
            # pd.DataFrame(data).to_csv(path_out + data_filename + ".csv", header=["t"] + systems[sys_name].data_column_names, index=False)

            # plot and save noisy data plots
            if make_plots and snr=="inf":
                plot_filename = f"fig_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}_{{}}.{{}}"
                plot_noisy_data(data, save_plot=True, plot_path=path_out, plot_name=plot_filename, num_plots=2)

            # numerically add derivatives:
            X = np.array(data[:, 1:])
            dX = np.array([np.gradient(Xi, sim_step) for Xi in X.T]).transpose()
            data_with_der = pd.DataFrame(np.hstack((data, dX)))
            data_with_der.to_csv(path_out + data_filename + ".csv",
                                 header=['t'] + \
                                        systems[sys_name].data_column_names +
                                        ['d' + systems[sys_name].data_column_names[i] for i in range(len(systems[sys_name].data_column_names))],
                                 index=False)

    print("---- \nFinished: " + sys_name)
