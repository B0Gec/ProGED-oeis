import numpy as np
import pandas as pd
from src.generate_data.add_noise_to_data import add_noise_to_data, plot_noisy_data
import os

version = "v1"
path_in = "D:\\Experiments\\MLJ23\\data\\other\\ode-strogatz-master\\"

model_names = ["d_vdp", "d_bacres", "d_barmag", "d_lv", "d_glider", "d_predprey", "d_shearflow"]

for model_name in model_names:
    new_model_name = model_name[2:]
    path_out= f"D:\\Experiments\\MLJ23\\data\\strogatz\\{new_model_name}\\"
    os.makedirs(path_out, exist_ok=True)

    data1 = np.array(pd.read_csv(path_in + model_name + "1.txt"))
    data2 = np.array(pd.read_csv(path_in + model_name + "2.txt"))
    times = np.tile(np.arange(0, 10, 0.1), 4).reshape(400, -1)

    data_joined = np.hstack((times, data1[:, 1:], data1[:, 0].reshape(400, -1), data2[:, 0].reshape(400, -1)))

    for i in range(4):
        plot_filename = f"{new_model_name}_init{i}_{{}}.{{}}"
        plot_noisy_data(data_joined[i*100:((1+i)*100), :3], save_plot=True, plot_path=path_out, plot_name=plot_filename, num_plots=3)

        data_joined_pd_der = pd.DataFrame(data_joined[i*100:((1+i)*100), :], columns=["t", "x", "y", "dx", "dy"])
        data_joined_pd_der.to_csv(path_out + new_model_name + str(i) + "_withder.csv", index=False)

        data_joined_pd = pd.DataFrame(data_joined[i*100:((1+i)*100), :3], columns=["t", "x", "y"])
        data_joined_pd.to_csv(path_out + new_model_name + str(i) + ".csv",  index=False)

