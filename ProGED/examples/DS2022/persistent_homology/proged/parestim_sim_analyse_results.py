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
data_version = 'all'
exp_version = 'e4'
structure_version = 's0'
analy_version = 'a1'
plot_version = 'p1'
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in = f"{path_main}{os.sep}results{os.sep}proged{os.sep}parestim_sim{os.sep}{exp_version}{os.sep}"

method = "proged"
exp_type = "parestim_sim"
sys_names = list(systems.keys())
snrs = ['inf', 30, 20, 13, 10, 7]
set_obs = "all"
n_inits = 4
sim_step = 0.1
sim_time = 10
data_length = int(sim_time/sim_step)

# get full table of results
results = pd.read_csv(path_base_in + f"overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
results['durationMin'] = results.duration / 60


# --------------------
##### plot time ######
# --------------------

# time vs noise, grouped by obs
g = sns.catplot(x="snr", y="duration", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.6, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / s", xlabel="SNR / dB")
plt.title("A", loc='left', pad=-30, fontweight="bold")
g.savefig(path_base_in + f"plot_duration_allSystems_{exp_version}_{analy_version}_{plot_version}.png", dpi=300)
plt.close()

g = sns.catplot(x="snr", y="durationMin", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.6, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / min", xlabel="SNR / dB")
plt.title("A", loc='left', pad=-30, fontweight="bold")
g.savefig(path_base_in + f"plot_durationMin_allSystems_{exp_version}_{analy_version}_{plot_version}.png", dpi=300)
plt.close()

# USED: vs noise, grouped by obs, separated by systems (more specific)
g = sns.catplot(x="snr", y="durationMin", hue="obs_type", col="system", col_wrap=5, data=results, kind="point", sharey=False,
                legend=False, legend_out=True, scale=0.6, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / min", xlabel="SNR / dB")
g.fig.suptitle("Duration based on observability for specific systems")
g.fig.subplots_adjust(top=.9)
g.savefig(path_base_in + f"plot_durationMin_perSystem_{exp_version}_{analy_version}_{plot_version}.png", dpi=300)
plt.close()

# ----------------------------
##### plot error - rmse (calucalated by DE) ######
# ----------------------------

# rmse vs noise, grouped by obs
#g = sns.catplot(x="noise", y="rmse", hue="obs_type", data=results, kind="point", sharey=False, legend=False)
#g.set(yscale='log')
#plt.legend(title="Observability")

g = sns.catplot(x="noise", y="rmse", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(yscale='log', ylabel="RMSE (by DE)", xlabel="SNR / dB")
plt.title("B", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + analy_version + "_plot" + plot_version + "_joined_RMSE.png", dpi=300)
plt.close()

# vs noise, grouped by obs, separated by systems (more specific)
#g = sns.catplot(x="noise", y="rmse", hue="obs_type", col="system", data=results, kind="point", sharey=False, legend=False)
#g.set(yscale='log')
#plt.legend(title = "Observability")

# g = sns.catplot(x="obs_type", y="rmse", hue="noise", data=results, sharey=False, kind="point")
# g.set(yscale='log')

# ----------------------------
##### plot error - trajectory error ######
# ----------------------------

g = sns.catplot(x="noise", y="TExy", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(yscale='log', ylabel="Trajectory Error", xlabel="SNR / dB")
plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + analy_version + "_plot" + plot_version + "_joined_TExy.png", dpi=300)
plt.close()

#g = sns.catplot(x="obs_type", y="TExy", hue="noise", data=results, kind="point", sharey=False, legend=False)
#g.set(yscale='log')
#plt.legend(title="Noise")

#g = sns.catplot(x="noise", y="TExy", hue="obs_type", col="system",data=results, sharey=False, kind="point", legend=False,
#                scale=0.8, errwidth=2, dodge=0.1, capsize=0.05)
#g.set(yscale='log')
#plt.legend(title="Observability")

# ----------------------------
##### plot error - reconstruction error ######
# ----------------------------

#g = sns.catplot(x="noise", y="RE", hue="obs_type", data=results, sharey=False, kind="point", legend=False,
#                scale=0.8, errwidth=2, dodge=0.1, capsize=0.05)
#plt.legend(title = "Observability")

g = sns.catplot(x="noise", y="RE", hue="obs_type", data=results, sharey=False, kind="point", legend=False, legend_out=True,
                scale=0.8, errwidth=1.5, dodge=0.1, capsize=0.05)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Reconstruction Error", xlabel="SNR / dB")
plt.title("D", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + analy_version + "_plot" + plot_version + "_joined_RE.png", dpi=300)
plt.close()

#g = sns.catplot(x="obs_type", y="RE", hue="noise", data=results, sharey=False, kind="point", legend=False)
#plt.legend(title="SNR")


