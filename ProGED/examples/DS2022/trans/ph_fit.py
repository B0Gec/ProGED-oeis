import pandas as pd
import numpy as np
import os
import warnings
import sys
import itertools as it

# import ProGED as pg
from ProGED import EqDisco, ModelBox, fit_models

# warnings.filterwarnings("ignore", message="dgm1 has points with non-finite death times;ignoring those points")
# warnings.filterwarnings("ignore", message="dgm2 has points with non-finite death times;ignoring those points")
warnings.filterwarnings("ignore", message="ODEintWarning: Excess work done on this call (perhaps wrong Dfun type). Run with full_output = 1 to get quantitative information.")
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=ODEintWarning)
 # model: [C0*y, C1*x**2*y + C2*x + C3*y]
# /home/breaky/Documents/py-envs/bulleyed/lib/python3.9/site-packages/persim/bottleneck.py:66: UserWarning: dgm2 has points with non-finite death times;ignoring those points
 #  warnings.warn(
# /home/breaky/Docu
# y, C1*x**2*y + C2*x + C3*y]
# /home/breaky/Documents/py-envs/bulleyed/lib/python3.9/site-packages/scipy/integrate/_odepack_py.py:247: ODEintWarning: Excess work done on this call (perhaps wrong Dfun type). Run with full_output = 1 to get quantitative information.
#   warnings.warn(warning_msg, ODEintWarning)
# /home/breaky/Documents/py


dirn = 'data/allonger/lorenz/'
dirn = 'data/allonger/myvdp/'
fname = dirn + 'data_lorenz_allonger_len2000_snrinf_init0.csv'
fname = dirn + 'data_myvdp_allonger_len2000_snrinf_init0.csv'
csv = pd.read_csv(fname)
# print(os.path.isdir(dirn))
# print(os.path.isfile(fname))
# print(os.getcwd())
# print(os.listdir())
# print(csv[['t', 'x', 'y', 'z']])
# print(csv[['t', 'x', 'y', 'z']].to_numpy())
data = csv[['t', 'x', 'y']].to_numpy()


# generation_settings = {"simulation_time": 0.25}
# data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8], **generation_settings)
# data = data[:, (0, 1, 2)]  # y, 2nd column, is not observed

job_id = int(sys.argv[1])
# job_id = 43
# combs = list(it.product(['x', 'y', 'xy'], [i for i in range(20)]))
combs = list(it.product(['x', 'y', 'xy'], [2, 3, 4, 5, 6, 8, 10, 15, 20]))
obstring, s = combs[job_id]
obs = [v for v in obstring]

print(combs)
print(obs, s)
# print(job_id, obsinput, sin)
# 1/0

max_iter = 2000
pop_size = 60

# s = 1 .. 20
# s = 4
max_iter = 100*s
pop_size = 3*s

np.random.seed(1)
# obs_combs = ['x', 'y']
# obs = ['x', 'y']
# obs = ['x']
# obs = ['y']
# obs = [var for var in obsinput]

print('s:', s, 'max_iter:', max_iter, 'pop_size:', pop_size)
print(obs)
print('')

data = csv[['t'] + obs].to_numpy()
# # obs xy
# if obs == 'xy':
#     system = ModelBox(observed=["x", "y"])
# # obs x
# elif obs == 'x':
#     system = ModelBox(observed=["x"])
# # obs y
# elif obs == 'y':
#     system = ModelBox(observed=["y"])
system = ModelBox(observed=obs)
system.add_system(["C*y", "C*y - C*x*x*y - C*x"], symbols={"x": ["x", "y"], "const": "C"})
# estimation_settings = {"target_variable_index": None,
#                        "time_index": 0,
                       # "optimizer": 'DE_scipy',
                       # "objective_settings": {"use_jacobian": False,
                       #                        "persistent_homology": True,
                       #                        },
                       # "optimizer_settings": {"max_iter": 1,
                       #                        "pop_size": 1},
                       # "verbosity": 1,

objective_settings = {
    "atol": 10 ** (-6),
    "rtol": 10 ** (-4),
    "use_jacobian": False,
    "teacher_forcing": False,
    "simulate_separately": False,
    "persistent_homology": True,
    # "persistent_homology_weight": 0.5,
    # "persistent_homology_size": 100, # ali 2000
    }

optimizer_settings = {
        "lower_upper_bounds": (-10, 10),
        "default_error": 10 ** 9,
        "strategy": 'best1bin', # best1bin
        "f": 0.5 , # 0.8
        "cr": 0.5,
        "max_iter": max_iter,
        "pop_size": pop_size,
        "atol": 0.001,
        "tol": 0.001,
        "pymoo_min_f": 10**(-5),
    }

estimation_settings = {
        "target_variable_index": None,
        "time_index": 0,
        "max_constants": 10,
        "optimizer": 'DE_pymoo',
        "observed":  obs,
        "optimizer_settings": optimizer_settings,
        "objective_settings": objective_settings,
        "default_error": 10 ** 9,
        "timeout": np.inf,
        "verbosity": 2,
    }

np.random.seed(1)
system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)


# print(f"All iters (as saved to system_model object): {system_out[0].all_iters}")
# print(f"Iters when PH was used: {system_out[0].ph_used}")
# print(f"Iters when zero vs zero: {system_out[0].zerovszero}")
# # print(abs(system_out[0].get_error()))
# assert abs(system_out[0].get_error()) < 1  # 3.2.2023
# true params: [[1.], [-0.5., -1., 0.5]]



# import itertools
# import numpy as np


# def get_fit_settings(obs):
#     objective_settings = {
#         "atol": 10 ** (-6),
#         "rtol": 10 ** (-4),
#         "use_jacobian": False,
#         "teacher_forcing": False,
#         "simulate_separately": False,
#         "persistent_homology": True,
#         "persistent_homology_weight": 0.5,
#         "persistent_homology_size": 100, # ali 2000
#         }

#     optimizer_settings = {
#         "lower_upper_bounds": (-10, 10),
#         "default_error": 10 ** 9,
#         "strategy": 'best1bin', # best1bin
#         "f": 0.5 , # 0.8
#         "cr": 0.5,
#         "max_iter": 2000,
#         "pop_size": 60,
#         "atol": 0.001,
#         "tol": 0.001,
#         "pymoo_min_f": 10**(-5),
#     }

#     estimation_settings = {
#         "target_variable_index": None,
#         "time_index": 0,
#         "max_constants": 10,
#         "optimizer": 'DE_pymoo',
#         "observed":  obs,
#         "optimizer_settings": optimizer_settings,
#         "objective_settings": objective_settings,
#         "default_error": 10 ** 9,
#         "timeout": np.inf,
#         "verbosity": 0,
#     }

#     return estimation_settings


# def get_opt_params(idx):
#     fs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     crs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#     all_combinations = list(itertools.product(fs, crs))
#     return all_combinations[idx][0], all_combinations[idx][1]


# np.random.seed(3)
# ED = EqDisco(data = data,
#              task = None,
#              task_type = "differential",
#              time_index = 0,
#              target_variable_index = 1,
#              variable_names=["t", "x", "y", ],
#              sample_size = 8,
#              verbosity = 3,
#              estimation_settings={
#                 "persistent_homology": True,
#                 # "persistent_homology_size": 200,
#                 # "persistent_homology_weight": 0.5,
#                 },
#              )
# ED.generate_models()
# ED.fit_models()

print('2 -- all done')
 

# system = ModelBox(observed=["x", "y", "z"])
# system.
# system.add_system(["C*(y-x)", "x*(C-z) - y", "x*y - C*z"], symbols={"x": ["x", "y", "z"], "const": "C"})
# estimation_settings = {"target_variable_index": None,
#                        "time_index": 0,
#                        "optimizer": 'DE_scipy',
#                        "optimizer_settings": {"max_iter": 1,
#                                               "pop_size": 1,
#                                               "lower_upper_bounds": (-28, 28),
#                                               },
#                        "objective_settings": {"use_jacobian": False,
#                                               "persistent_homology": True,
#                                               },
#                        "verbosity": 2,
#                        }

# np.random.seed(0)
# system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)
# print(f"All iters (as saved to system_model object): {system_out[0].all_iters}")
# print(f"Iters when PH was used: {system_out[0].ph_used}")
# print(f"Iters when zero vs zero: {system_out[0].zerovszero}")
# # print(abs(system_out[0].get_error()))
# assert abs(system_out[0].get_error()) < 1.0  # 3.2.2023

print('s:', s, 'max_iter:', max_iter, 'pop_size:', pop_size)
print(obs)
print('')
print('EOF')
