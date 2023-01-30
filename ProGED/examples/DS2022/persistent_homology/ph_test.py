# all vs all:
import os
import time
import numpy as np
# import pickle as pkl
import pandas as pd
import ProGED as pg
import ipywidgets

# Basic imports 
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import networkx as nx
# from IPython.display import Video

# scikit-tda imports..... Install all with -> pip install scikit-tda
#--- this is the main persistence computation workhorse
import ripser
# from persim import plot_diagrams
import persim
# import persim.plot
import teaspoon.TDA.Draw as Draw

import sys
# sys.executable

def big_loada():
    """Loads the 5 lorenz trajectories into a list."""

    traj_names = ["lorenzA", "lorenzB", "lorenzC", "lorenzD", "lorenzE"]
    lorenzs = []
    for name in traj_names:
        place = "~/ProGED/ProGED/examples/DS2022/persistent_homology/data/"
        lorenz_pd = pd.read_csv(place + name + '_data.csv')
        lorenz = lorenz_pd.to_numpy()[:, 1:]
        lorenzs += [lorenz]
        if not np.array([np.shape(i) == lorenzs[0].shape for i in lorenzs]).all():
            # if not (lorenzA.shape == lorenzB.shape and lorenzB.shape == lorenzE.shape):
            print("point clouds of different shapes!!!!!!")
    return lorenzs


# size = 100000 # 4-8sec
size = 1000  # 4-8sec
size = 500  # 4-8sec
# size = 200  # 4-8sec

def traj2diag(traj, size):
    """Returns p diagram of given trajectory"""

    def downsample(lorenz):
        m = int(lorenz.shape[0] / size)
        lorenz = lorenz[:(m * size), :]
        def aggregate(array):
            return array.reshape(-1, m).mean(axis=1)
        lor = np.apply_along_axis(aggregate, 0, lorenz)
        return lor

    if size < traj.shape[0]:
        P1 = downsample(traj)
    diagrams1 = ripser.ripser(P1)['dgms']
    return diagrams1, P1

# 3d scatter tested:
# debug oneliner:
# P1 = trajectory; import matplotlib.pyplot as plt; fig = plt.figure(); ax = fig.add_subplot(projection='3d').scatter(P1[:, 0], P1[:, 1], P1[:, 2], s=1); plt.show()

# 3d scatter tested good:
def plottr(P1: np.ndarray):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    # ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], marker=".")
    ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], s=1)
    plt.show()

# # # archive code:
# # # plot original input trajectories dim=2: (press "q" to quit plot)
# # for P1 in lorenzs:
# #     print('now')
# #     plt.scatter(P1[:, 0], P1[:, 1], label='P1')
# #     plt.show()
# #
# # ## plot full res dim=3:
# # ## no scatter, dots connected with lines, press "q" to quit plot.
# # for P1 in lorenzs:
# #     plt.close()
# #     fig = plt.figure()
# #     ax = plt.axes(projection='3d')
# #     ax.plot3D(P1[:, 0], P1[:, 1], P1[:, 2])
# #     plt.show()
# #     print('plot3d')



if __name__ == "__main__":

    downs = []
    diags = []

    traj_names = ["lorenzA", "lorenzB", "lorenzC", "lorenzD", "lorenzE"]
    lorenzs = big_loada()
    for lorenz in lorenzs:
        diagrams1, P1 = traj2diag(lorenz, size)
        downs += [P1]
        diags += [diagrams1]


    # for P1 in lorenzs[:1]:
    # (press "q" to quit plot)
    for P1 in lorenzs:
        plottr(P1)

    plt.close('all')


    # dists = []
    # plot diagrams:
    for diag in diags:
        for i in diag:
            print(i[:20])
        persim.plot_diagrams(diag, show=True)
        # distance_bottleneck, matching = persim.bottleneck(diag[1], diagrams3[1], matching=True)
        # persim.bottleneck_matching(diagrams1[1], diagrams3[1], matching=matching, labels=['Clean $H_1$', 'Noisy $H_1$'])
        # Draw.drawDgm(diag[1])
        print('diag plotted')

    dists = []
    for i in range(len(diags)):
        for j in range(i+1, len(diags)):
            plt.close()
            distance_bottleneck, matching = persim.bottleneck(diags[i][1], diags[j][1], matching=True)
            dists += [(i, j, distance_bottleneck)]
            print(f'distance lorenz {i} vs lorenz {j} = {distance_bottleneck}')
            print(f'diagram of bottleneck distance lorenz {i} vs lorenz {j}:')
            mse = np.mean((lorenzs[i]-lorenzs[j])**2)
            persim.bottleneck_matching(diags[i][1], diags[j][1], matching=matching,
                                               labels=[f'{traj_names[i]} {distance_bottleneck}',
                                                       f'{traj_names[j]} {mse}'])
            # persim.visuals.bottleneck_matching(diags[i][1], diags[j][1], matching=matching,
            #                                    labels=[f'{traj_names[i]} {distance_bottleneck}',
            #                                            f'{traj_names[j]} {mse}'])
            results = 'ph_dists/'
            # results = place + 'ph_dists/'
            #### plt.sav#e#fig(f'{results}ph_dist_{traj_names[i]}-vs-{traj_names[j]}-{round(distance_bottleneck,2)} .png', dpi=300)
            plt.show()
            # plt.close()


                #
    # A vs D: 3.5

    #
    # plt.scatter(P1[:, 0], P1[:, 1], label='P1')
    # # plt.show()
    # plt.scatter(P2[:, 0], P2[:, 1], label='P1')
    # # plt.show()
    # plt.scatter(P3[:, 0], P3[:, 1], label='P1')
    # # lorenzApd
    # # lorenzBpd
    #
    #
    # distance_bottleneck, matching = persim.bottleneck(diagrams1[1], diagrams2[1], matching=True)
    # distance_bottleneck1v3, matching = persim.bottleneck(diagrams1[1], diagrams3[1], matching=True)
    # # persim.visuals.bottleneck_matching(diagrams1[1], diagrams2[1], matching=matching, labels=['lorenzA $\dot{y}=f(x,y)$', 'LorenzE $H_1$'])
    # persim.visuals.bottleneck_matching(diagrams1[1], diagrams3[1], matching=matching, labels=['lorenzA $\dot{y}=f(x,y)$', 'LorenzB $H_1$'])
    # print('The bottleneck distance is', distance_bottleneck)
    # print('The bottleneck distance p1 v p3 is', distance_bottleneck1v3)
    # # diagrams1
    # # diagrams2
    #

    print('end')

