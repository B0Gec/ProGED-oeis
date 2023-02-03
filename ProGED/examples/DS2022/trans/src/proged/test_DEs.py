import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tabulate import tabulate
import ProGED as pg
from src.generate_data.systems_collection import strogatz, mysystems
from src.generate_data.system_obj import System
from plotting.parestim_sim_plots import plot_trajectories, plot_optimization_curves

# experiment types
exps = ["test_DEscipy1", "test_DEpymoo1", "test_DEmanual1"]
system = "bacres"

# get true data
data_filepath = f"D:\\Experiments\\MLJ23\\data\\strogatzl\\{system}\\"
data_filename = f"data_{system}_strogatzl_len100_snrinf_init0.csv"
itr_true = np.array(pd.read_csv(data_filepath + data_filename))

# get results
results = []

for exp in exps:
    results_filepath = f"D:\\Experiments\\MLJ23\\results\\proged\\parestim_sim\\{exp}\\{system}\\"
    results_filename = f"{system}_strogatzl_s0_{exp}_len100_snrinf_init0_obsxy_fitted"
    systemBox = pg.ModelBox()
    systemBox.load(results_filepath + results_filename + ".pg")

    results.append(systemBox)
    #systemBox.dump(results_filepath + results_filename+ "_edited.pg")

# evaluation
print(f"DE Errors:\n"
      f"{exps[0]}: {results[0][0].estimated.fun} \n" 
      f"{exps[1]}: {results[1][0].estimated['fun'][0]} \n" 
      f"{exps[2]}: {results[2][0].estimated['fun']} \n")

print(f"Duration:\n"
      f"{exps[0]}: {results[0][0].estimated.time} \n" 
      f"{exps[1]}: {results[1][0].estimated['time']} \n" 
      f"{exps[2]}: {results[2][0].estimated['time']} \n")

print(f"Parameters:\n"
      f"{exps[0]}: {results[0][0].estimated.x} \n" 
      f"{exps[1]}: {results[1][0].estimated['x']} \n" 
      f"{exps[2]}: {results[2][0].estimated['x']} \n")
