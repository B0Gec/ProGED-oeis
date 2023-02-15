import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
x_0, x_1, x_2 = sp.symbols("x_0,x_1,x_2")

# system = [0.8400669308169996*(0.03951635028682724*x_0 + 0.03450190344652977*(0.93150133750629448 - 0.0158337713316539*x_1)*(3.88719086192104*x_1 - 155.03257835779529) - 1)**2 - 0.8, -0.03241879963527478*x_0 - 0.07655628438854671*x_1 + (0.0185255124580351*x_1 + 0.092143435117635539)*(0.5726691332851931*x_0 - 0.0011799999999999998*sin(0.07215631079393435*x_0 - 48.42598621254726) - 11.271054067835362) + 10.486893486623094]
#
systems = ["0.8400669308169996*(0.03951635028682724*x_0 + 0.03450190344652977*(0.93150133750629448 - 0.0158337713316539*x_1)*(3.88719086192104*x_1 - 155.03257835779529) - 1)**2 - 0.8",
          "-0.03241879963527478*x_0 - 0.07655628438854671*x_1 + (0.0185255124580351*x_1 + 0.092143435117635539)*(0.5726691332851931*x_0 - 0.0011799999999999998*sin(0.07215631079393435*x_0 - 48.42598621254726) - 11.271054067835362) + 10.486893486623094"]
#
# print(system)
# print(systems)

# df[]



eq = x_0*x_1 + 1
print(eq)
print(str(eq))

import pickle as p
# file = open('pik.pk', 'wb')
# p.dump(eq, file)
# file.close()

import  os
print(os.getcwd())

### console:
## os.chdir("ProGED/examples/DS2022/trans")
readfile = "results/ete/lorenz/eqs_lorenz_all_len100_snrinf_init1_obsxyz_595183.179094531"
# file = open('pik.pk', 'rb')
file = open(readfile, 'rb')
model = p.load(file)
file.close()
print(model)


# ete | cphase | snr: 13 | obs: xy | init: 1

# 1/0
from src.generate_data.system_obj import System

orig_expr = ["20 - x - ((x * y) / (1 + 0.5 * x**2))",
           "10 - (x * y / (1 + 0.5 * x**2))"]
# vdp:
orig_expr = ["10 * (y - (1 / 3 * (x ^ 3 - x)))",
             "- 1/10*x"]
orig_expr = ["-5 * (x - 1 / 3 * x ** 3 - y)",
             "-1/5*x"]
n_inits = 5
mbacr = System(sys_name="bacres",
                benchmark="strogatz",
                sys_vars=["x", "y"],
                # orig_expr=["20 - x - ((x * y) / (1 + 0.5 * x**2))",
                #            "10 - (x * y / (1 + 0.5 * x**2))"],
                # sym_structure=["C - (x*y / (C * x**2 + C)) - x",
                #                "C - (x*y / (C * x**2 + C))"],
               # sym_structure=["20 - (x*y / (0.5 * x**2 + 1)) - x",
               #                "10 - (x*y / (0.5 * x**2 + 1))"],
               sym_structure = orig_expr,
               # sym_structure = systems,
               # sym_params=[[20, 0.5, 1], [10, 0.5, 1]],
                inits=np.tile([5, 10], n_inits).reshape(n_inits, -1) + np.random.normal(1, 1, (n_inits, 2)),
                # bounds=[0, 20],
                # data_column_names=["x", "y"],
                )

traj = mbacr.simulate(0, sim_time=1000, sim_step=0.1)
print(traj[:6, :])
# 1/0

mbacr = System(sys_name="etetest",
               benchmark="strogatz",
               # sys_vars=["x", "y"],
               sys_vars=["x_0", "x_1"],
               # orig_expr=["20 - x - ((x * y) / (1 + 0.5 * x**2))",
               #            "10 - (x * y / (1 + 0.5 * x**2))"],
               # sym_structure=["C - (x*y / (C * x**2 + C)) - x",
               #                "C - (x*y / (C * x**2 + C))"],
               sym_structure = [str(model[0]),
                                str(model[1])],
               sym_params=[[20, 0.5, 1], [10, 0.5, 1]],
               inits=np.tile([5, 10], n_inits).reshape(n_inits, -1) + np.random.normal(1, 1, (n_inits, 2)),
               )
def plot3d(P1: np.ndarray, title=""):
    plt.close()
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    # ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], marker=".")
    if P1.shape[1] == 3:
        ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], s=1)
    elif P1.shape[1] == 2:
        ax.scatter(P1[:, 0], P1[:, 1], s=1)
    elif P1.shape[1] == 1:
        ax.scatter(P1[:, 0], np.zeros((P1.shape[0], 1)), s=1)
    else:
        raise IndexError
    plt.show()

plot3d(traj[:, 1:], 'systems')
# plot3d(traj[:, 0:], 'systems')
