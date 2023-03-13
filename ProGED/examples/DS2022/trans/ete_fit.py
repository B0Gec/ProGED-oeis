import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ProGED as pg
import time
import itertools
import pickle as p
# import pysindy as ps
from scipy.integrate import odeint
### console:
import os
# sys.path.append("ProGED/examples/DS2022/trans")
# os.chdir("ProGED/examples/DS2022/trans")
# os.chdir("../../../")
# os.chdir("..")
# os.path.abspath("__file__")
# print(sys.path)
#
# # sys.path.append(os.path.join(os.path.dirname(__file__),'path/to/module')
# # sys.path.append(os.path.join(os.getcwd(), 'path/to/module'))
# # sys.path.append(os.getcwd())
# # os.chdir("ProGED/examples/DS2022/trans")
#
# print(os.getcwd())

from src.generate_data.systems_collection import strogatz, mysystems
from src.generate_data.system_obj import System
# from ProGED.examples.DS2022.trans.src.generate_data.systems_collection import strogatz, mysystems
from ProGED.examples.DS2022.trans.mlj import ete

np.random.seed(1)

def plot3d(P1: np.ndarray, savename, title="", showoff=False):
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
    # plt.show()
    plt.savefig(savename, dpi=300)
    if showoff:
        plt.show()
    return 0



DEFAULT_SETTINGS = {
    'max_input_points': 300,
    'n_trees_to_refine': 100,
}

def ete_fit(X, x_dot: np.ndarray, sys_name="default", settings=DEFAULT_SETTINGS):
    # print(X.shape, x_dot.shape)
    eqs = []
    for i in range(x_dot.shape[1]):
        y = x_dot[:, i]
        # print(y.shape)
        eqs.append(ete(X, y, settings))
    # print(eqs)
    return eqs

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
# data_version = "allonger"
data_version = "all"
set_obs = "full"  # either full, part or all
# snrs = ["inf", 30, 20, 13, 10, 7]
snrs = ['inf', '30', '13']
inits = np.arange(0, 4)
loaded_models = False

timestamp = time.perf_counter()
if mode == 'debug':
    timestamp = "debug"
elif mode == 'slurm':
    timestamp = "slurm"
    if len(sys.argv) >= 2:
        job = int(sys.argv[1])
    if len(sys.argv) >= 3:
        data_version = sys.argv[2]
    if len(sys.argv) >= 4:
        expname = sys.argv[3]
        timestamp += expname

elif mode == 'paralel':
    timestamp = 'exp1'
    timestamp = 'exp3'
    if len(sys.argv) >= 2:
        timestamp = sys.argv[1]
    if timestamp == 'exp1':
        # from src.generate_data.systems_collection import strogatz
        # strogatz = {'vdp': strogatz['vdp']}
        strogatz = {}
        # mysystems = {'lorenz': mysystems['lorenz']}
        mysystems = {'lorenz': mysystems['lorenz']}
        systems = {**strogatz, **mysystems}
    # elif timestamp == 'exp2':
    # strogatz = {'': strogatz['vdp']}
print(f'timestamp: {timestamp}')

if mode == 'debug':
    snrs = ['inf']
    # snrs = ['inf', '30']
    # inits = np.array([0])
    # inits = np.array([0, 1, 2])
    strogatz = {'lv': strogatz['lv']}
    # strogatz = {'vdp': strogatz['vdp']}
    mysystems = {}
    # loaded_models = True
elif mode == 'paralel':
    # 13.24
    # snrs = ['13']
    # snrs = ['30']
    # 13.22 [inf 30]
    # snrs = ['inf', '30']
    # snrs = ['inf', ]
    # 13.27 all snr
    # inits = np.array([1])
    # data_pts = 1850
    # # inits = np.array([3])
    # mysystems = {}
    # mysystems = {'lorenz': mysystems['lorenz']}
    # # mysystems = {'cphase': mysystems['cphase']}
    # # # mysystems = {'stl': mysystems['stl']}
    # # strogatz = {}
    # strogatz = {
    #         # 'bacres': strogatz['bacres'],
    #         #     'barmag': strogatz['barmag'],
    #             'glider': strogatz['glider'],
    #             'lv': strogatz['lv'],
    #             'predprey': strogatz['predprey'],
    #             'shearflow': strogatz['shearflow'],
    # #             'vdp': strogatz['vdp'],
    #             }
    pass
if mode == 'slurm':
    # strogatz = {}
    # strogatz = {'lv': strogatz['lv']}
    # strogatz = {'vdp': strogatz['vdp']}
    # mysystems = {}
    # mysystems = {'myvdp': mysystems['myvdp']}
    # snrs = ['inf']
    # snrs = ['30', '13']
    # inits = np.array([0])
    pass

settings = {
    'max_input_points': 4000,
    # 'n_trees_to_refine': 1,
    'n_trees_to_refine': 3,
}
print(settings)



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
if mode == 'slurm':
    if len(combinations) > job:
        combinations = [combinations[job]]
    else:
        combinations = []
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
path_main = "D:\\Experiments\\MLJ23"
path_main = "."


path_base_out = f"{path_main}{os.sep}results{os.sep}{method}_{timestamp}{os.sep}"
os.makedirs(path_base_out, exist_ok=True)
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
df_mold = tuple(np.nan for _ in range(5))

start_global = time.time()

count_error_fit, count_error_simulate = 0, 0
for sys_name, iobs, iinit, snr in combinations:
    iobs_name = ''.join(iobs)
    print(f"{method} | {sys_name} | snr: {snr} | obs: {''.join(iobs)} | init: {iinit}")


    key = f"{sys_name}_{data_version}_len{data_length}" \
          f"_snr{snr}_init{iinit}_obs{iobs_name}"

    if key in ete_df.columns:
        continue

    path_out = path_base_out + sys_name + os.sep
    os.makedirs(path_out, exist_ok=True)

    # case of debugging:
    path_pickle = f"{path_main}{os.sep}results{os.sep}{method}_saved{os.sep}"
    pkl_filename = path_pickle + sys_name + os.sep + "eqs_" + key + "eq.pkl"

    # Get true clean data
    path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
    data_true_inf = pd.read_csv(path_data_in + f"data_{sys_name}_{data_version}_len{data_length}_snrinf_init{iinit}.csv")
    # print('po csv')
    data_true_inf = np.array(data_true_inf[['t'] + systems[sys_name].data_column_names])
    # Get true noisy data
    data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
    data_orig = pd.read_csv(path_data_in + data_filename)
    data = np.array(data_orig[['t'] + systems[sys_name].data_column_names])
    col_names = list(data_orig.columns[:np.shape(data)[1]])
    # get derivatives
    data_der = np.array(data_orig[['d' + systems[sys_name].data_column_names[i] for i in range(len(systems[sys_name].data_column_names))]])

    X = data[:, 1:]
    # chpase:
    # X = data[:, :]
    t = data[:, 0]
    x0 = data[0, 1:]


    entry = list(df_mold)

    startTime = time.time()

    # if not loaded_models or not os.path.isfile(pkl_filename):

    model = ete_fit(X, x_dot=data_der, sys_name=sys_name, settings=settings)
    print(model)
    # else:
    #     file = open(pkl_filename, 'rb')
    #     model = p.load(file)
    #     file.close()
    #     print('model successfully pickled')

    entry[2:2 + len(model)] = [eq for eq in model]
    pass
    # except Exception as e:
    #     print(e)
    #     count_error_fit += 1
    #     # eqs_df[key] = [np.nan, np.nan] + [np.nan for i in range(X.shape[1])] + [np.nan]
    #     # results_list.append(ires)
    #     print('Error in sindy_fit. Continue with next system')

    # check duration
    endTime = time.time()
    duration = endTime - startTime
    duration_global = endTime - start_global
    print("duration:", duration, 'overall duration:', duration_global)

    ete_df[key] = entry
    # eqs_df[key] = [np.nan, np.nan] + [np.nan for i in range(X.shape[1])] + [np.nan]
    # eqs_df[key] = [eq for eq in model]

    entry[0] = duration
    ete_df[key] = entry
    csv_filename = f"{path_base_out}{key}.csv"
    ete_df.to_csv(csv_filename, index=False)

    continue

    # save eqs
    path_out = path_base_out + sys_name + os.sep
    os.makedirs(path_out, exist_ok=True)
    # out_filename = path_out + f"eqs_{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}" \
    #                                f"_snr{snr}_init{iinit}_obs{iobs_name}"
    # key = f"{sys_name}_{data_version}_len{data_length}" \
    #            f"_snr{snr}_init{iinit}_obs{iobs_name}{{}}.{{}}"
    out_filename = path_out + key
    # print(key)
    try:
    # print(out_filename)
        file = open(out_filename + "eq.pkl", 'wb')
        p.dump(model, file)
        file.close()
    except Exception as e:
        print(f'unsuccessful pickle: {e}')


    # ma = pg.model(model[0], sym_vars=['x_0', 'x_1'])
    # mb = pg.model(model[1], sym_vars=['x_0', 'x_1'])
    # mbo = pg.ModelBox.models_dict[str(model[0])] = ma

    valid_sim = True
    # # simulate
    try:
        n_inits = 5
        sym_vars_lib = ["x_0", "x_1", "x_2", "x_3", "x_4",]
        # # ijmodel = System(sys_name, sym_vars_lib[:X.shape[1]])

        system = System(sys_name="anonymous",
                       benchmark="strogatz",
                       # sys_vars=["x", "y"],
                       sys_vars=sym_vars_lib[:X.shape[1]],
                       # orig_expr=["20 - x - ((x * y) / (1 + 0.5 * x**2))",
                       #            "10 - (x * y / (1 + 0.5 * x**2))"],
                       # sym_structure=["C - (x*y / (C * x**2 + C)) - x",
                       #                "C - (x*y / (C * x**2 + C))"],
                       # sym_structure=["20 - (x*y / (0.5 * x**2 + 1)) - x",
                       #                "10 - (x*y / (0.5 * x**2 + 1))"],
                       # sym_structure = orig_expr,
                       sym_structure=model,
                       # sym_params=[[20, 0.5, 1], [10, 0.5, 1]],
                       inits=x0.reshape(1, -1)
                       # bounds=[0, 20],
                       # data_column_names=["x", "y"],
                       )

        sim_step = t[1]-t[0]
        sim_time = t[-1] + sim_step
        # print(sim_time, sim_step)
        sim = system.simulate(0, sim_time=sim_time, sim_step=sim_step)

        # sym_vars_lib = ["x_0", "x_1", "x_2", "x_3", "x_4",]
        # # ijmodel = System(sys_name, sym_vars_lib[:X.shape[0]])
        # sys_func = [expr.lambdify() for expr in model]
        # X = data[:, 1:]
        # t = data[:, 0]
        # x0 = data[0, 1:]
        # sim_time, sim_step = t[-1], t[1]-t[0]
        #
        # def custom_func(t, x):
        #     return [sys_func[i](*x) for i in range(len(sys_func))]
        #
        # simulation = odeint(custom_func, x0, t, rtol=1e-12, atol=1e-12, tfirst=True)
        # TEx = np.sqrt((np.mean((simulation[:, 0] - data[:, 1]) ** 2))) / np.std(data[:, 1])
        # TEy = np.sqrt((np.mean((simulation[:, 1] - data[:, 2]) ** 2))) / np.std(data[:, 2])

    # # #     sim = model.simulate(x0, t, integrator="odeint")
    # # traj = mbacr.simulate(0)
    except Exception as e:
        valid_sim = False
        print(f'{e}')
        print('Error in simulating model. Continue with next system')
        continue

    if len(sim) != len(t) or np.any(np.isnan(sim)):
        print(sys_name, snr, iinit, "Simulation failed due to size mismatch or nans in simulation.")
        valid_sim = False
        continue

    # save simulation
    if valid_sim:
        # path_out = path_base_out + sys_name + os.sep
        # os.makedirs(path_out, exist_ok=True)
        # out_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}" \
        #                f"_snr{snr}_init{iinit}_obs{iobs_name}_{{}}.{{}}"
        # sim_with_t = np.hstack((t.reshape(t.size, 1), sim))
        sim_with_t = sim

        simulation = pd.DataFrame(sim_with_t, columns=col_names)
        simulation.to_csv(out_filename + "_simulation.csv", index=None)

    # trajectory error
    if not valid_sim:
        count_error_simulate += 1
        # TExs = [np.nan for i in range(1, X.shape[1] + 1)]
        # TExyz = np.nan

    # TEx = np.sqrt((np.mean((sim_with_t[:, 1] - data[:, 1]) ** 2))) / np.std(data[:, 1])
    # TEy = np.sqrt((np.mean((sim_with_t[:, 2] - data[:, 2]) ** 2))) / np.std(data[:, 2])
    # TExy = TEx + TEy
    TExs = [np.sqrt((np.mean((sim_with_t[:, i] - data[:, i]) ** 2))) / np.std(data[:, i]) for i in range(1, X.shape[1]+1)]
        # TEx_i = np.sqrt((np.mean((sim_with_t[:, i] - data[:, i]) ** 2))) / np.std(data[:, i])
    TExyz = sum(TExs)

    entry[1] = TExyz
    ete_df[key] = entry
    # ete_df[key] = ete_df[key][:1] + [TExyz] + ete_df[key][1:]
    # ete_df.to_csv(path_out + "ete.csv", index=False)
    ete_df.to_csv(csv_filename, index=False)

    # ete_extended[key] = entry[:2] + TExs + entry[2:]

    # tex = sgrt ( mean( ))



    # # plot phase trajectories in a single figure and save
    if plot_trajectories_bool and valid_sim:
        plot_filepath = f"{path_out}plots{os.sep}"
        # print(plot_filepath)
        os.makedirs(plot_filepath, exist_ok=True)
        plot_filename = plot_filepath + key
        showoff = True
        showoff = False
        # print(plot_filename)
        plot3d(data[:, 1:], plot_filename + "_no_time_truth.png", key + " without time, ground truth", showoff=showoff)
        plot3d(sim[:, 1:], plot_filename + "_no_time.png", key + " without time", showoff=showoff)
        if sim_with_t.shape[1] <= 3:
            plot3d(data, plot_filename + "_with_time_truth.png" , key + " with time, ground truth", showoff=showoff)
            plot3d(sim_with_t, plot_filename + "_with_time.png", key + " with time", showoff=showoff)


print(ete_df)

print('after loop')
print(f'Errors: count_error_simulate = {count_error_simulate}, count_error_fit = {count_error_fit}')

print(f'{timestamp}', systems)
print("finished")