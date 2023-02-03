# Code to analyse and plot results of the (only) parameter estimation of various dynamical systems.

import seaborn as sns
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

systems = {**strogatz, **mysystems}
method = 'proged'
exp_type = 'parestim_num'
data_version = 'all'
exp_version = 'e1'
structure_version = 's0'
analy_version = 'a1'
plot_version = 'p1'
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
path_out = f"{path_base_in}analysis{os.sep}"
base_plot_name = f"_{method}_{exp_type}_{exp_version}_{analy_version}_{plot_version}.png"
os.makedirs(path_out, exist_ok=True)


sys_names = list(systems.keys())
snrs = ['inf', 30, 20, 13, 10, 7]
set_obs = "full"
n_inits = 4
sim_step = 0.1
sim_time = 10
data_length = int(sim_time/sim_step)

# get full table of results
results = pd.read_csv(path_base_in + f"overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
results['durationMin'] = results.duration / 60
# results['rmseDE'] = [float(results['rmseDE'][i][1:-1]) for i in range(len(results['rmseDE']))]


# --------------------
##### plot time ######
# --------------------

# time vs noise, grouped by obs
g = sns.catplot(x="snr", y="duration", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.3, errwidth=1.2, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / s", xlabel="SNR / dB")
# plt.title("A", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_duration_allSystems_{base_plot_name}", dpi=300)
plt.close()

g = sns.catplot(x="snr", y="durationMin", hue="obs_type", data=results, sharey=False, kind="point",
                legend=False, legend_out=True, scale=0.6, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / min", xlabel="SNR / dB")
# plt.title("A", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_durationMin_allSystems_{base_plot_name}", dpi=300)
plt.close()

# USED: vs noise, grouped by obs, separated by systems (more specific)
plot_version = 'p1'
g = sns.catplot(x="snr", y="durationMin", hue="obs_type", col="system", col_wrap=5, data=results, kind="point", sharey=True,
                legend=False, legend_out=True, scale=0.6, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / min", xlabel="SNR / dB")
g.fig.suptitle("Duration based on observability for specific systems")
g.fig.subplots_adjust(top=.9)
g.savefig(path_out + f"plot_durationMin_perSystem_{base_plot_name}", dpi=300)
plt.close()

# ----------------------------
##### plot error - rmse (calucalated by DE) ######
# ----------------------------

# rmse vs noise, grouped by obs, per system
plot_version = 'p1'
g = sns.catplot(x="snr", y="rmseDE", hue="obs_type", data=results, col="system", col_wrap=5,
                sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="RMSE (by DE)", xlabel="SNR / dB") # yscale='log',
g.savefig(path_out + f"plot_DErmse_perSystems_{base_plot_name}", dpi=300)
plt.close()

# rmse vs noise, grouped by obs, grouped by all systems
plot_version = 'p1'
g = sns.catplot(x="snr", y="rmseDE", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(yscale='log', ylabel="RMSE (by DE)", xlabel="SNR / dB")
plt.title("B", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_DErmse_allSystems_{base_plot_name}", dpi=300)
plt.close()

# ----------------------------
##### plot error - trajectory error ######
# ----------------------------
plot_version = 'p1'
g = sns.catplot(x="snr", y="TExy", hue="obs_type", data=results, sharey=False,
                legend=False, legend_out=True, col="system", col_wrap=5,
                kind="point", scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylim=(0, None), ylabel="Trajectory Error", xlabel="SNR / dB") # yscale='log'
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_TExy_perSystems_{base_plot_name}", dpi=300)
plt.close()

plot_version = 'p1'
g = sns.catplot(x="snr", y="TExy", hue="obs_type", data=results, sharey=False,
                legend=False, legend_out=True,
                kind="point", scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylim=(0, None), ylabel="Trajectory Error", xlabel="SNR / dB") # yscale='log'
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_TExy_allSystems_{base_plot_name}", dpi=300)
plt.close()

# ----------------------------
##### plot error - reconstruction error ######
# ----------------------------

plot_version = 'p1'
g = sns.catplot(x="snr", y="RE", hue="obs_type", data=results, sharey=False, kind="point",
                legend=False, legend_out=True, col="system", col_wrap=5,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Reconstruction Error", xlabel="SNR / dB") # ylim=(0, 10),
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_RE_perSystems_{base_plot_name}", dpi=300)
plt.close()


plot_version = 'p1'
g = sns.catplot(x="snr", y="RE", hue="obs_type", data=results, sharey=False, kind="point",
                legend=False, legend_out=True,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Reconstruction Error", xlabel="SNR / dB") # ylim=(0, 10),
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_RE_allSystems_{base_plot_name}", dpi=300)
plt.close()
