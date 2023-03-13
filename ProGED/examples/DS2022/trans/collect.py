### This file collects transformers results from slurm jobs,
### one csv per task / one job per task.

##
import pandas as pd
import os
##
# b = pd.read_csv('./results/good'+"ete.csv")
# fn = 'results/ete_792097.771365873/ete.csv'
# a = pd.read_csv(fn)
##

# files = ['']
names = [
    'lorenz',
    'cphase',
    'stl',
    'bacres',
    'barmag',
    'glider',
    'lv',
    'predprey',
    'shearflow',
    'vdp',
    ]

# final = pd.DataFrame()
# predir = '../../'
# dir = 'results/good/'
# os.chdir('../../')
# # dir = predir + 'results/good/'
# for name in names:
#     fn = dir + name + "100.csv"
#     print(os.getcwd())
#     print(os.path.isfile(fn))
#     df = pd.read_csv(fn)
#     print(df)
#     final = pd.concat([final, df], axis=1)
#     print(final)
# # final.to_csv(dir + "ete100.csv", index=False)
#


import os
import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import ProGED as pg
# import time
import itertools
# import pickle as p
# # import pysindy as ps
# from scipy.integrate import odeint
# ### console:
# import os
# # sys.path.append("ProGED/examples/DS2022/trans")
# # os.chdir("ProGED/examples/DS2022/trans")
# # os.chdir("../../../")
# # os.chdir("..")
# # os.path.abspath("__file__")
# # print(sys.path)
# #
# # # sys.path.append(os.path.join(os.path.dirname(__file__),'path/to/module')
# # # sys.path.append(os.path.join(os.getcwd(), 'path/to/module'))
# # # sys.path.append(os.getcwd())
# # # os.chdir("ProGED/examples/DS2022/trans")
# #
# # print(os.getcwd())

from src.generate_data.systems_collection import strogatz, mysystems
# from src.generate_data.system_obj import System
# # from ProGED.examples.DS2022.trans.src.generate_data.systems_collection import strogatz, mysystems
# from ProGED.examples.DS2022.trans.mlj import ete
#
# np.random.seed(1)

# def plot3d(P1: np.ndarray, savename, title="", showoff=False):
#     plt.close()
#     fig = plt.figure()
#     plt.title(title)
#     ax = fig.add_subplot(projection='3d')
#     # ax = plt.axes(projection='3d')
#     # ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], marker=".")
#     if P1.shape[1] == 3:
#         ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], s=1)
#     elif P1.shape[1] == 2:
#         ax.scatter(P1[:, 0], P1[:, 1], s=1)
#     elif P1.shape[1] == 1:
#         ax.scatter(P1[:, 0], np.zeros((P1.shape[0], 1)), s=1)
#     else:
#         raise IndexError
#     # plt.show()
#     plt.savefig(savename, dpi=300)
#     if showoff:
#         plt.show()
#     return 0
#


DEFAULT_SETTINGS = {
    'max_input_points': 300,
    'n_trees_to_refine': 100,
}


mode = 'debug'
mode = 'paralel'
mode = 'slurm'


## MAIN
print('till main')
method="ete"
exp_type="sysident_num"
exp_version = "e2"
analy_version = 'a1'
structure_version = "s0"
data_version = "allonger"
data_version = "all"
set_obs = "full"  # either full, part or all
# snrs = ["inf", 30, 20, 13, 10, 7]
snrs = ['inf', '30', '13']
inits = np.arange(0, 4)
loaded_models = False

timestamp = "ete_slurmnina"
timestamp = "slurm1"

# # timestamp = time.perf_counter()
# if mode == 'debug':
#     timestamp = "debug"
# elif mode == 'slurm':
#     timestamp = "slurm"
#     if len(sys.argv) >= 2:
#         job = int(sys.argv[1])
#     if len(sys.argv) >= 3:
#         data_version = sys.argv[2]
#     if len(sys.argv) >= 4:
#         expname = sys.argv[3]
#         timestamp += expname
#



systems = {**strogatz, **mysystems}
# systems = {**mysystems}
# systems = {**strogatz}
if data_version == "allonger":
    data_length = 2000
else:
    data_length = 100
# data_length = 1000
print(f'data_length: {data_length}')
sindy_threshold = 0.001
plot_trajectories_bool = True

sys_names = list(systems.keys())
combinations = []
for sys_name in sys_names:
    combinations.append(list(itertools.product([sys_name], systems[sys_name].get_obs(set_obs), inits, snrs)))
    print([sys_name], systems[sys_name].get_obs(set_obs), inits, snrs)
combinations = [item for sublist in combinations for item in sublist]
# if mode == 'slurm':
#     if len(combinations) > job:
#         combinations = [combinations[job]]
#     else:
#         combinations = []
print('combinations', combinations)
print('script inputs:', sys.argv)


def job2task(job_id):
    counter = 0
    for system in systems:
        for init in inits:
            for snr in snrs:
                if job == counter:
                    print(system, init, snr)
    return 0
print('till combo')
# 1/0
path_main = "."


path_base_out = f"{path_main}{os.sep}results{os.sep}{method}_{timestamp}{os.sep}"
# os.makedirs(path_base_out, exist_ok=True)
print(path_base_out)

ete_df = pd.DataFrame()
if not mode == 'slurm':
    # previous experiments checkpoint
    csv_filename = f"{path_base_out}ete.csv"
    if os.path.isfile(csv_filename):
        ete_df = pd.read_csv(csv_filename)
print(f'cases this far: {ete_df.columns}')

# ete_extended = pd.DataFrame()
## [duration, TExyz, eq_dx, eq_dy, eq_dz]  # explained:


inside = []
final = pd.DataFrame()
count_error_fit, count_error_simulate = 0, 0
for sys_name, iobs, iinit, snr in combinations[:]:
    iobs_name = ''.join(iobs)
    print(f"{method} | {sys_name} | snr: {snr} | obs: {''.join(iobs)} | init: {iinit}")
    # 1/0


    #
    # if key in ete_df.columns:
    #     continue
    #
    # path_out = path_base_out + sys_name + os.sep
    # os.makedirs(path_out, exist_ok=True)
    #
    # # case of debugging:
    # path_pickle = f"{path_main}{os.sep}results{os.sep}{method}_saved{os.sep}"
    # pkl_filename = path_pickle + sys_name + os.sep + "eqs_" + key + "eq.pkl"
    #
    # # Get true clean data
    # path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
    # data_true_inf = pd.read_csv(path_data_in + f"data_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv")
    # # print('po csv')
    # data_true_inf = np.array(data_true_inf[['t'] + systems[sys_name].data_column_names])
    # # Get true noisy data
    # data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
    # data_orig = pd.read_csv(path_data_in + data_filename)
    # data = np.array(data_orig[['t'] + systems[sys_name].data_column_names])
    # col_names = list(data_orig.columns[:np.shape(data)[1]])
    # # get derivatives
    # data_der = np.array(data_orig[['d' + systems[sys_name].data_column_names[i] for i in range(len(systems[sys_name].data_column_names))]])
    #
    # X = data[:, 1:]
    # # chpase:
    # # X = data[:, :]
    # t = data[:, 0]
    # x0 = data[0, 1:]
    #
    #
    # entry = list(df_mold)
    #
    # startTime = time.time()
    #
    # # if not loaded_models or not os.path.isfile(pkl_filename):
    #
    #
    # # check duration
    # endTime = time.time()
    # duration = endTime - startTime
    # duration_global = endTime - start_global
    # print("duration:", duration, 'overall duration:', duration_global)
    #
    # ete_df[key] = entry
    # # eqs_df[key] = [np.nan, np.nan] + [np.nan for i in range(X.shape[1])] + [np.nan]
    # # eqs_df[key] = [eq for eq in model]
    #
    # entry[0] = duration
    # ete_df[key] = entry
    #

    path_main = "."
    path_main = "../.."
    detail = "good/refine3/"
    path_base_out = f"{path_main}{os.sep}results{os.sep}{detail}" \
                    f"{timestamp}{os.sep}"
    key = f"{sys_name}_{data_version}_len{data_length}" \
          f"_snr{snr}_init{iinit}_obs{iobs_name}"
    csv_filename = f"{path_base_out}{key}.csv"
    # csv_filename = "../.." + csv_filename

    # print(csv_filename)
    # print(path_base_out)
    # print(os.path.isdir(path_base_out))
    # # print(os.listdir(path_base_out))
    # # print("../../")
    # print(os.path.isfile(csv_filename))

    if os.path.isfile(csv_filename):
        df = pd.read_csv(csv_filename)
        final = pd.concat([final, df], axis=1)
        print(df)
        inside += [f"{method} | {sys_name} | snr: {snr} | obs: {''.join(iobs)} | init: {iinit}"]
    # print(os.getcwd())
save = "../../results/good/refine3/ete2000re3n.csv"
# final.to_csv(save, index=False)
print(final)

for i in inside:
    print(i)







## old collect file:
# bacres | snr: 13 | obs: xy | init: 1
# 1.43


# 2000
# /results/ete_8037.495613274/
# ete | barmag | snr: 30 | obs: xy | init: 0
# ete_13023.425707316/ ete | barmag | snr: 13 | obs: xy | init: 0
# ./results/ete_14444.361517777/ ete | barmag | snr: 13 | obs: xy | init: 1
# vdp snr13 init 1 10.57





# 1500
# 111sec bacres
# 1600 10.53-
# 1700 10.53-
# 1800 300sec


# lorenz  inf init 2  7.07
# 801200.264429347 {'lorenz': []
# sults/ete_796518.769197929/
# lorenz [3] ['inf', '30', '13']
# ./results/ete_801359.161109003/


# lv      inf init 3  7.07
# te_798442.891162022/


# 100:
# 796464.088333189 {'cphase'
# stl100.csv

# bacres
# 799246.874708268 {'barmag':
# 798487.135007086 {'glider':
# 800992.431140143 {'lv':
# ete_797113.74386133/predprey_ete.csv
# e_796557.288991034/ ete | shearflow
# ete_796993.69314761/ ete | vdp, cphase, stl

# data_length = 2000
#  9.37
# old vdp:
# gth: 2000
# try ['vdp'] [['x', 'y']] [0 1 2 3] ['inf', '30', '13']
# till combo
# ./results/ete_368778.911807757/
# success ['vdp'] [0 1 2] ['inf', '30', '13'] + [3] ['inf']

# length: 100
# ['vdp'] [['x', 'y']] [0 1 2 3] ['inf', '30', '13']
# till combo
# ./results/ete_434299.246681834/




print('EOF')

# (-0.00134 - 4.62/((0.0922 - 0.00037999999999999997*cos(97.366794289155003*x_1 - 7.470686355837214))*(0.0007378537362560646*x_0 + 68.992625151906121)*(0.003754310659551916*x_2 - 6.9600382693747702)*(2.111799745997953*x_2 - 0.5125265233082643)*(-0.001370052360043538*x_0 + 0.086337832464639727*x_1 - 19.788345768054894*x_2 - 26.007657183938502)))*(15.8 - 0.030000000000000002/(0.037000000000000005 - 75.60000000000001*arctan(8.302582507377336 - 730.5262825044771*x_2))),
# (0.0022961550214962497 - 0.03030179410371872*x_1)*(0.001110244706432247*x_0 - 0.058096895840790322) + (0.98926445456258167*x_1 + 0.002037291945269509)*(-0.001329651691924637*x_2 - 0.09998644626310221)]

# (8.7657350358699793e-5 - 0.31445021911613402*x_1)*(-0.0094*Abs(1.50172668985322*sqrt(sqrt(sqrt(0.859470468431772*Abs(-53.53827431995441*x_0 - (-6.856400253509147*x_0 - 69.177934955738679)*(46.32065288619398*x_1 - 0.34675093053293236) + 1.0243461966788003) + 1) + 0.003979641386895956) - 0.2949187914922992) - 22.1) - 3.0100000000000002)

## vdp refine3:
# 0.3566595126574321*x_2 + (0.19006197713981578*x_2 - 0.0019373870977437824)*(0.358893560403193*(0.000676089091809518 - exp(79.10031973064686*(0.008815366963966533 - x_2)**2*(-sqrt(-0.014846625766871167 + 1/(4.23*Abs(0.0086894850738605142*x_1 - 0.03485845621939966) + 2.19)) - 0.20793029211910517)**2))**2 - 0.00401) - 0.141635590603174,
# (1.1400000000000001 + 0.00182/(-1215.458076029933*x_2 + (0.20136647103397652 - 3.2195656235201139*x_1)*(110.27648440847784 - 11.03316502335946*x_0) + (0.02927166230687205*x_0 - 0.61657026475718612)*(0.004555294193317958*x_0 - 45.36458713625232*x_2 + 0.79499144634494497) + 7.2576800818676458))*(-0.086115025155053571*x_1 - 4.4530030821318834e-5)

## vdp snr 30 i0 ref3:
# [(0.0002568676356982009*x_1 + 0.0001796780482572927)*(56.16984999978205*cos(6.9011342803872136*x_0 + 458.04775151479919*x_1 - 57.261186476374378) + 1.8861495599405205),
# 0.248*tan(-0.056609244028413372*x_0 + 0.18371878191132704*x_1 + 59.799473971346108) - 0.0021]
# (0.11001297961807544*x_0 + 0.0007787276251901708)*(0.007020790544562112*x_1 - 0.90014048755075955)]
# (0.11001297961807544*x_0 )*(- 0.90014048755075955)]






