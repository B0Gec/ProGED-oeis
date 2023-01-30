import matplotlib.pyplot as plt
import persim

from ph_test import traj2diag, plottr, big_loada

if __name__ == "__main__":

    # size = 100000 # 4-8sec
    size = 1000  # 4-8sec
    size = 500  # 4-8sec
    size = 200  # 4-8sec

    lorenzs = big_loada()
    traj1 = lorenzs[0]
    traj2 = lorenzs[1]
    traj2 = lorenzs[2]


    print("input trajectory:")
    plottr(traj1)
    diagrams1, P1 = traj2diag(traj1, size)
    print("downsampled trajectory:")
    plottr(P1)
    print("diagram:")
    persim.plot_diagrams(diagrams1, show=True)

    print("second trajectory:")
    plottr(traj2)
    diagrams2, P2 = traj2diag(traj2, size)
    print("downsampled second trajectory:")
    plottr(P2)
    print("second diagram:")
    persim.plot_diagrams(diagrams2, show=True)

    print("bottleneck distance diagram:")
    distance_bottleneck, matching = persim.bottleneck(diagrams1[1], diagrams2[1], matching=True)
    print("bottleneck distance = ", distance_bottleneck)
    persim.bottleneck_matching(diagrams1[1], diagrams2[1], matching=matching, labels=['traj1', 'second trajectory'])
    plt.show()

    print('end')

