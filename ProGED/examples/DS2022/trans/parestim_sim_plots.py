import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectories(trajectories_true, trajectories_model, trajectories_true_inf, save_figs=True,
                      fig_path="", fig_name="", plot_single_all=False, plot_one_example=False):

    # Plot phase plot
    if plot_single_all:
        for ip in range(len(trajectories_model)):
            plt.figure()
            if np.size(trajectories_true[0], 1) < 4:
                plt.plot(trajectories_true[ip][:, 1], trajectories_true[ip][:, 2], color='black', linewidth=2)
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], color='red', linewidth=1, linestyle="--")
                plt.legend(['true', 'model'])
            else:
                plt.plot(trajectories_true[ip][:, 1], trajectories_true[ip][:, 2], color='black', linewidth=2)
                plt.plot(trajectories_true[ip][:, 3], trajectories_true[ip][:, 4], color='gray', linewidth=2)
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], color='red', linewidth=1, linestyle="--")
                plt.plot(trajectories_model[ip][:, 3], trajectories_model[ip][:, 4], color='orange', linewidth=1, linestyle="--")
            plt.legend(['true1', 'true2', 'model1', 'model2'])
            plt.xlabel("X")
            plt.ylabel("Y")
            #plt.xlim([-10, 10])
            #plt.ylim([-10, 10])
            if save_figs:
                plt.savefig(fig_path + fig_name.format("phaseplot" + str(ip), "png"), dpi=300)

            # Plot time series

            # for non coupled systems
            if np.size(trajectories_true[0], 1) < 4:
                fig, ax = plt.subplots(2)
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 1], color='black', linewidth=2, linestyle="--")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 2], color='black', linewidth=2, linestyle="--")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 1], color='red', linewidth=1, linestyle="--")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 2], color='red', linewidth=1, linestyle="--")
                legend_entries = ['true', 'model']
                plt.legend(legend_entries, loc='center left', bbox_to_anchor=(0.8, 0.8))
                ax[0].set(xlabel='time', ylabel='X')
                ax[1].set(xlabel='time', ylabel='Y')

            # for coupled systems
            else:
                fig, ax = plt.subplots(4)
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 1], color='black', linewidth=2, linestyle="--")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 1], color = 'red', linewidth=1, linestyle="--")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 2], color='black', linewidth=2, linestyle="--")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 2],  color = 'red', linewidth=1, linestyle="--")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 3], color='gray', linewidth=2, linestyle="--")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 3],  color = 'orange', linewidth=1, linestyle="--")
                ax[3].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 4], color='gray', linewidth=2, linestyle="--")
                ax[3].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 4],  color = 'orange', linewidth=1, linestyle="--")
                legend_entries = ['trueX1', 'modelX1', 'trueY1' 'modelY1', 'trueX2', 'modelX2', 'trueY2' 'modelY2']
                plt.legend(legend_entries, loc='center left', bbox_to_anchor=(0.8, 0.8))
                ax[0].set(xlabel='', ylabel='X1')
                ax[1].set(xlabel='', ylabel='Y1')
                ax[2].set(xlabel='', ylabel='X2')
                ax[3].set(xlabel='time', ylabel='Y2')

            if save_figs:
                plt.savefig(fig_path + fig_name.format("timeplot" + str(ip), "png"), dpi=300)
                plt.close('all')
            else:
                plt.show()
    elif plot_one_example:
        # plot just one example:
        for ip in range(1):

            # for non coupled:
            if np.size(trajectories_true[0], 1) == 3:
                plt.figure()
                plt.plot(trajectories_true_inf[ip][:, 1], trajectories_true_inf[ip][:, 2], color='blue', linewidth=1, linestyle="-")
                plt.plot(trajectories_true[ip][:, 1], trajectories_true[ip][:, 2], color='black', linewidth=0.5, linestyle="-")
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], color='red', linewidth=1, linestyle="-")
                plt.legend(['true inf', 'true', 'model'])

            # for lorenz
            elif np.size(trajectories_true[0], 1) == 4:
                plt.figure().add_subplot(projection='3d')
                plt.plot(trajectories_true_inf[0][:, 1], trajectories_true_inf[0][:, 2], trajectories_true_inf[0][:, 3], color='blue', linewidth=1, linestyle="-", label="true_inf")
                plt.plot(trajectories_true[0][:, 1], trajectories_true[0][:, 2], trajectories_true[0][:, 3], color='black', linewidth=0.5, linestyle="-", label="true")
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], trajectories_model[ip][:, 3], color='red', linewidth=1, linestyle="-", label="model")
                plt.legend(['true inf', 'true', 'model'])
            # for coupled:
            else:
                plt.figure()
                plt.plot(trajectories_true[ip][:, 1], trajectories_true[ip][:, 2], color='black', linewidth=2)
                plt.plot(trajectories_true[ip][:, 3], trajectories_true[ip][:, 4], color='gray', linewidth=2)
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], color='red', linewidth=1, linestyle="--")
                plt.plot(trajectories_model[ip][:, 3], trajectories_model[ip][:, 4], color='orange', linewidth=1, linestyle="--")
                plt.legend(['true1', 'true2', 'model1', 'model2'])
            plt.xlabel("X")
            plt.ylabel("Y")
            # plt.xlim([-10, 10])
            # plt.ylim([-10, 10])
            if save_figs:
                plt.savefig(fig_path + fig_name.format("phaseplot" + str(ip), "png"), dpi=300)

            # Plot time series for one example
            if np.size(trajectories_true[0], 1) == 3:
                fig, ax = plt.subplots(2)
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true_inf[ip][:, 1], color='blue', linewidth=1, linestyle="-")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 1], color='black', linewidth=0.5, linestyle="-")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 1], color='red', linewidth=1, linestyle="-")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true_inf[ip][:, 2], color='black', linewidth=1, linestyle="-")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 2], color='black', linewidth=0.5, linestyle="-")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 2], color='red', linewidth=1, linestyle="-")
                legend_entries = ['true_inf', 'true', 'model']
                plt.legend(legend_entries, loc='center left', bbox_to_anchor=(0.8, 0.8))
                ax[0].set(xlabel='time', ylabel='X')
                ax[1].set(xlabel='time', ylabel='Y')

            elif np.size(trajectories_true[0], 1) == 4:
                fig, ax = plt.subplots(3)
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true_inf[ip][:, 1], color='blue', linewidth=1, linestyle="-")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 1], color='black', linewidth=0.5, linestyle="-")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 1], color='red', linewidth=1,linestyle="-")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true_inf[ip][:, 2], color='blue', linewidth=1, linestyle="-")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 2], color='black', linewidth=0.5, linestyle="-")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 2], color='red', linewidth=1,linestyle="-")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_true_inf[ip][:, 3], color='blue', linewidth=1, linestyle="-")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 3], color='black', linewidth=0.5, linestyle="-")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 3], color='red', linewidth=1, linestyle="-")
                legend_entries = ['true_inf', 'true', 'model']
                plt.legend(legend_entries, loc='center left', bbox_to_anchor=(0.8, 0.8))
                ax[0].set(xlabel='time', ylabel='X')
                ax[1].set(xlabel='time', ylabel='Y')
                ax[1].set(xlabel='time', ylabel='Z')

            else:
                fig, ax = plt.subplots(4)
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 1], color='black', linewidth=2, linestyle="--")
                ax[0].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 1], color = 'red', linewidth=1, linestyle="--")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 2], color='black', linewidth=2, linestyle="--")
                ax[1].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 2],  color = 'red', linewidth=1, linestyle="--")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 3], color='gray', linewidth=2, linestyle="--")
                ax[2].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 3],  color = 'orange', linewidth=1, linestyle="--")
                ax[3].plot(trajectories_true[ip][:, 0], trajectories_true[ip][:, 4], color='gray', linewidth=2, linestyle="--")
                ax[3].plot(trajectories_true[ip][:, 0], trajectories_model[ip][:, 4],  color = 'orange', linewidth=1, linestyle="--")
                legend_entries = ['trueX1', 'modelX1', 'trueY1' 'modelY1', 'trueX2', 'modelX2', 'trueY2' 'modelY2']
                plt.legend(legend_entries, loc='center left', bbox_to_anchor=(0.8, 0.8))
                ax[0].set(xlabel='', ylabel='X1')
                ax[1].set(xlabel='', ylabel='Y1')
                ax[2].set(xlabel='', ylabel='X2')
                ax[3].set(xlabel='time', ylabel='Y2')

            if save_figs:
                plt.savefig(fig_path + fig_name.format("timeplot" + str(ip), "png"), dpi=300)
                plt.close('all')
            else:
                plt.show()
    else:
        # plot all initial values together
        if np.size(trajectories_true[0], 1) == 3:
            plt.figure()
            for ip in range(len(trajectories_model)):
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], linewidth=1.5, linestyle="--", label="model")
            plt.plot(trajectories_true[0][:, 1], trajectories_true[0][:, 2], color='black', linewidth=0.5, label="true")
            plt.plot(trajectories_true_inf[0][:, 1], trajectories_true_inf[0][:, 2], color='red', linewidth=0.5, label="true_inf")
            plt.legend(frameon=False)

        elif np.size(trajectories_true[0], 1) == 4:
            plt.figure().add_subplot(projection='3d')
            for ip in range(len(trajectories_model)):
                plt.plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], trajectories_model[ip][:, 3], linewidth=1.5, linestyle="--", label="model")
            plt.plot(trajectories_true[0][:, 1], trajectories_true[0][:, 2], trajectories_true[0][:, 3], color='black', linewidth=0.5, label="true")
            plt.plot(trajectories_true_inf[0][:, 1], trajectories_true_inf[0][:, 2], trajectories_true_inf[0][:, 3], color='red', linewidth=0.5, label="true_inf")
            plt.legend(frameon=False)

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            axes[0].plot(trajectories_true[0][:, 1], trajectories_true[0][:, 2], color='black', linewidth=1)
            axes[1].plot(trajectories_true[0][:, 3], trajectories_true[0][:, 4], color='black', linewidth=1)
            axes[0].plot(trajectories_true_inf[0][:, 1], trajectories_true_inf[0][:, 2], color='red', linewidth=0.5)
            axes[1].plot(trajectories_true_inf[0][:, 3], trajectories_true_inf[0][:, 4], color='red', linewidth=0.5)
            axes[0].legend(['true 1', 'true inf 1'], frameon=False)
            axes[1].legend(['true 2', 'true inf 2'], frameon=False)

            for ip in range(len(trajectories_model)):
                axes[0].plot(trajectories_model[ip][:, 1], trajectories_model[ip][:, 2], linewidth=1, linestyle="--")
                axes[1].plot(trajectories_model[ip][:, 3], trajectories_model[ip][:, 4], linewidth=1, linestyle="--")

        plt.title(fig_name.format("phaseplot", "png"))
        plt.xlabel("X")
        plt.ylabel("Y")

        if save_figs:
            plt.savefig(fig_path + fig_name.format("phaseplot", "png"), dpi=300)
        plt.close('all')


def plot_optimization_curves(systemBox, fig_path="", fig_name="", save_figs=True, optimizer="DE_pymoo"):
    for im in range(len(systemBox)):
        plt.figure()
        if optimizer == "DE_pymoo":
            optimization_curve_pd = pd.DataFrame(systemBox[im].estimated["all_results"].problem.opt_curve,
                                                 columns=["objective"])
        else:
            optimization_curve_pd = pd.DataFrame(systemBox[im].optimization_curve,
                                                 columns=["objective"])

        plt.plot(optimization_curve_pd.objective)
        plt.gca().set_ylim(bottom=0)
        plt.axhline(y=1, color='k', linestyle='--')

        if np.nanmax(np.array(optimization_curve_pd)[optimization_curve_pd != np.inf]) > 10 ** 3:
            plt.gca().set_ylim(top=10**3)

        plt.title(fig_name.format("convergence_curve", "png"))
        if save_figs:
            plt.savefig(fig_path + fig_name.format("convergence_curve", "png"), dpi=300)
        plt.close('all')