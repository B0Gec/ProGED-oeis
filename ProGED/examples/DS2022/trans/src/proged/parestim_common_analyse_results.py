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
exp_type = 'parestim_comparison'
data_version = 'all'
exp_version = 'e1'
structure_version = 's0'
analy_version = 'a1'
plot_version = 'p1'
base_plot_name = f"_{method}_{exp_type}_{exp_version}_{analy_version}"
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_in_num = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}parestim_num{os.sep}{exp_version}{os.sep}"
path_base_in_sim = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}parestim_sim{os.sep}{exp_version}{os.sep}"
path_out = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}analysis{os.sep}"
os.makedirs(path_out, exist_ok=True)


sys_names = list(systems.keys())
snrs = ['inf', 30, 20, 13, 10, 7]
set_obs = "full"
n_inits = 4
sim_step = 0.1
sim_time = 10
data_length = int(sim_time/sim_step)

# get full table of results
results_num = pd.read_csv(path_base_in_num + f"overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
results_num.drop('b', inplace=True, axis=1)
results_num['durationMin'] = results_num.duration / 60
results_num['exp_type'] = 'parestim_num'
results_sim = pd.read_csv(path_base_in_sim + f"overall_results_table_{data_version}_{structure_version}_{exp_version}_{analy_version}.csv", sep='\t')
results_sim['durationMin'] = results_sim.duration / 60
results_sim['rmseDE'] = [float(results_sim['rmseDE'][i][1:-1]) for i in range(len(results_sim['rmseDE']))]
results_sim['exp_type'] = 'parestim_sim'

results = pd.concat([results_num, results_sim])
results['system'] = results['system'].astype('category')
results['obs_type'] = results['obs_type'].astype('category')
results['snr'] = results['snr'].astype('category')
results['exp_type'] = results['exp_type'].astype('category')
results['expobs_type'] = results.exp_type.astype('str') + '_' + results.obs_type.astype('str')
results['expobs_type'] = results['expobs_type'].astype('category')

palette = sns.color_palette("dark")[:3] + sns.color_palette("bright")[:3] + sns.color_palette("pastel")[2:10]
markers = list(np.tile('o', 3)) + list(np.tile('s', 3)) + list(np.tile('v', 8))
linestyle = list(np.tile('--', 3)) + list(np.tile(':', 3)) + list(np.tile('-', 8))
hue_order = ['parestim_num_xy', 'parestim_num_xyuv', 'parestim_num_xyz',
             'parestim_sim_xy', 'parestim_sim_xyuv', 'parestim_sim_xyz',
             'parestim_sim_x', 'parestim_sim_y',
             'parestim_sim_yuv', 'parestim_sim_xuv',
             'parestim_sim_xyv', 'parestim_sim_xyu',
             'parestim_sim_yz', 'parestim_sim_xz']

# --------------------
##### plot time ######
# --------------------

# all systems - seconds
plot_version = 'p2'
g = sns.catplot(x="snr", y="duration", hue="expobs_type", data=results, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=False, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Exp type", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / s", xlabel="SNR / dB")
# plt.title("A", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_duration_allSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# all systems - minutes
g = sns.catplot(x="snr", y="durationMin", hue="expobs_type", data=results, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=True, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
g.set(ylabel="Time / min", xlabel="SNR / dB")
# plt.title("A", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_durationMin_allSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# per system - minutes
plot_version = 'p2'
g = sns.catplot(x="snr", y="durationMin", hue="expobs_type", col="system", col_wrap=5, data=results, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=False, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Time / min", xlabel="SNR / dB")
g.fig.suptitle("Duration based on observability for specific systems")
g.fig.subplots_adjust(top=.9)
g.savefig(path_out + f"plot_durationMin_perSystem_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# ---------------------------------------------------
######## plot error - rmse (calucalated by DE) ######
# -------------------------------------------------

# all systems
plot_version = 'p2'
g = sns.catplot(x="snr", y="rmseDE", hue="expobs_type", data=results,
                sharey=False, kind="point", hue_order=hue_order, legend=False, legend_out=True, scale=0.3,
                errwidth=0.2, dodge=0.2, capsize=0.05, markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="RMSE (by DE)", xlabel="SNR / dB") # yscale='log',
g.savefig(path_out + f"plot_DErmse_allSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# per system
plot_version = 'p2'
g = sns.catplot(x="snr", y="rmseDE", hue="expobs_type", data=results,  col="system", col_wrap=5, kind="point",
                hue_order=hue_order, legend=False, legend_out=True, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette, sharey=False)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(yscale='log', ylabel="RMSE (by DE)", xlabel="SNR / dB")
plt.title("B", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_DErmse_perSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# ----------------------------
##### plot error - trajectory error ######
# ----------------------------

# all systems
plot_version = 'p1'
g = sns.catplot(x="snr", y="TExy", hue="expobs_type", data=results, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=True, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylim=(0, 10), ylabel="Trajectory Error", xlabel="SNR / dB") # yscale='log'
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_TExy_allSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# per system
plot_version = 'p1'
g = sns.catplot(x="snr", y="TExy", hue="expobs_type", data=results, col="system", col_wrap=5, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=True, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylim=(0, 100), ylabel="Trajectory Error", xlabel="SNR / dB") # yscale='log'
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_TExy_perSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()

# ----------------------------
##### plot error - reconstruction error ######
# ----------------------------

plot_version = 'p2'
g = sns.catplot(x="snr", y="RE", hue="expobs_type", data=results, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=True, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Reconstruction Error", xlabel="SNR / dB") # ylim=(0, 10),
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_RE_allSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()


plot_version = 'p2'
g = sns.catplot(x="snr", y="RE", hue="expobs_type", data=results, col="system", col_wrap=5, sharey=False, kind="point",
                hue_order=hue_order, legend=False, legend_out=True, scale=0.3, errwidth=0.2, dodge=0.2, capsize=0.05,
                markers=markers, linestyles=linestyle, palette=palette)
plt.legend(title="Observability", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.set(ylabel="Reconstruction Error", xlabel="SNR / dB") # ylim=(0, 10),
# plt.title("C", loc='left', pad=-30, fontweight="bold")
g.savefig(path_out + f"plot_RE_perSystems_{base_plot_name}_{plot_version}.png", dpi=300)
plt.close()
