import numpy as np
import itertools


# set input when on hpc cluster
def set_numdiff_input(snrs, sys_names, n_inits, n_batches, iinput):

    inits_idx = np.arange(0, n_inits, 1)
    batches_idx = np.arange(0, n_batches, 1)

    all_combinations = list(itertools.product(*[sys_names, snrs, inits_idx, batches_idx]))
    current_job = list(all_combinations[iinput - 1])
    return current_job + get_system(current_job[0])

# set input when on hpc cluster
def set_sim_input(snrs, set_obs, n_inits, iinput):

    sys_names_short = ['vdp', 'stl', 'lotka', 'cphase']
    obs_short = ["xy", "x", "y"]
    sys_names_twostate = ["vdpvdpUS", "stlvdpBL"]
    obs_twostate = ["xyuv", "xyu", "xyv", "xuv", "yuv"]
    sys_names_lorenz = ["lorenz"]
    obs_lorenz = ["xyz", "xy", "xz", "yz"]

    inits_idx = np.arange(0, n_inits, 1)

    input_short = [sys_names_short, obs_short, snrs, inits_idx] if set_obs == "all" else [sys_names_short, [obs_short[0]], snrs, inits_idx]
    input_twostate = [sys_names_twostate, obs_twostate, snrs, inits_idx]  if set_obs == "all" else [sys_names_twostate, [obs_twostate[0]], snrs, inits_idx]
    input_lorenz = [sys_names_lorenz, obs_lorenz, snrs, inits_idx] if set_obs == "all" else [sys_names_lorenz, [obs_lorenz[0]], snrs, inits_idx]

    all_combinations = list(itertools.product(*input_short)) + list(itertools.product(*input_twostate)) + list(itertools.product(*input_lorenz))
    current_job = list(all_combinations[iinput - 1])

    return current_job + get_system(current_job[0], set_obs=current_job[1])

def get_system(sys_name, set_obs="full"):

    system_expressions = {
        "vdpOrig": ["y", "C*x + C*y*(1 - x**2)"],
        "vdp": ["y", "C*x**2*y + C*x + C*y"],
        "stlOrig": ["C*y + x*(C - x**2 - y**2)", "C*x + y*(C - x**2 - y**2)"],
        "stl": ["C*x + C*y - x**3 - x*y**2", "C*x + C*y - x**2*y - y**3"],
        "lotka": ["C*x*y + C*x", "C*x*y + C*y"],
        "cphase": ["C*sin(x) + C*sin(y) + C*sin(C*t) + C", "C*sin(x) + C*sin(y) + C"],
        "lorenzOrig": ["C*(-x + y)", "C*x - x*z - y", "C*z + x*y"],
        "lorenz": ["C*x + C*y", "C*x - x*z - y", "C*z + x*y"],
        "vdpvdpUSOrig":  ["y", "C*x + C*y*(1 - x**2)", "v", "C*u + C*v*(1 - u**2) + C*y**2"],
        "vdpvdpUS":  ["y", "C*x**2*y + C*x + C*y", "v", "C*u**2*v + C*u + C*v + C*y**2"],
        "stlvdpBLOrig": ["C*y + x*(C - x**2 - y**2)", "C*x + y*(C - x**2 - y**2) + C*v", "v", "C*u + C*v*(1 - u**2) + C*y"],
        "stlvdpBL": ["C*x + C*y - x**3 - x*y**2", "C*v + C*x + C*y - x**2*y - y**3", "v", "C*u**2*v + C*u + C*v + C*y"]
    }

    system_params = {
        "vdpOrig":[[], [-3., 2.]],               # freq w^2 = 3; eta = 2
        "vdp":[[], [-2., -3., 2.]],               # freq w^2 = 3; eta = 2
        "stlOrig":[[-3., 1.], [3., 1.]],         # freq w = 3; a = 1
        "stl":[[1., -3.], [3., 1.]],         # freq w = 3; a = 1
        "lotka":[[-0.02, 0.1], [0.02, -0.4]],
        "cphase":[[0.8, 0.8, -0.5, 2*np.pi*0.0015, 2.], [0, 0.6, 4.53]],
        "lorenzOrig":[[10.], [28.], [-8/3]],
        "lorenz":[[10., 10.], [28.], [-8/3]],
        "vdpvdpUSOrig": [[], [-3., 0.5], [], [-3., 0.5, 0.4]],         # freq w = 3; eta = 0.5; coupling C = 0.4
        "vdpvdpUS": [[], [0.5, -3., 0.5], [], [0.5, -3., 0.5, 0.4]],         # freq w = 3; eta = 0.5; coupling C = 0.4
        "stlvdpBLOrig": [[-3., 1.], [0.8, 3., 1.], [], [-3., 0.5, 0.4]], # freq w = 3; a = 1, eta = 0.5; couplings C1 = 0.8, C2 = 0.2
        "stlvdpBL": [[-3., 1.], [0.8, 3., 1.], [], [0.5, -3., 0.5, 0.4]], # freq w = 3; a = 1, eta = 0.5; couplings C1 = 0.8, C2 = 0.2
    }

    system_bounds = {
        "vdpOrig": (-5, 5),
        "vdp": (-5, 5),
        "stlOrig": (-5, 5),
        "stl": (-5, 5),
        "lotka": (10, 100),
        "cphase": (-5, 5),
        "lorenzOrig": (-5, 30),
        "lorenz": (-5, 30),
        "vdpvdpUSOrig": (-5, 5),
        "vdpvdpUS": (-5, 5),
        "stlvdpBLOrig": (-5, 5),
        "stlvdpBL": (-5, 5),
    }

    system_symbols = {
        "vdpOrig": ["x", "y"],
        "vdp": ["x", "y"],
        "stlOrig": ["x", "y"],
        "stl": ["x", "y"],
        "lotka": ["x", "y"],
        "cphase": ["x", "y", "t"],
        "lorenzOrig": ["x", "y", "z"],
        "lorenz": ["x", "y", "z"],
        "vdpvdpUSOrig": ["x", "y", "u", "v"],
        "vdpvdpUS": ["x", "y", "u", "v"],
        "stlvdpBLOrig": ["x", "y", "u", "v"],
        "stlvdpBL": ["x", "y", "u", "v"],
    }

    # observability scenarios
    if set_obs == "full":
        obs = [system_symbols[sys_name]]
        obstxt = ''.join(obs[0])
    elif set_obs == 'all':
        updated_symb = system_symbols[sys_name][:-1] if 't' in system_symbols[sys_name] else system_symbols[sys_name]
        obs = []
        for i in range(1, len(updated_symb) + 1):
            els = [list(x) for x in itertools.combinations(updated_symb, i)]
            obs.extend(els)
    else:
        obstxt = set_obs
        obs = [str(i) for i in obstxt]

    # set grammar type
    system_classification = {
        'onestate': ['vdp', 'stl', 'lotka'],
        'lorenz': ['lorenz'],
        'twophase': ['cphase'],
        'twostate1': ['vdpvdpUS', 'stlvdpBL'],
        'twostate2': ['vdpvdpUS', 'stlvdpBL']
    }

    grammar_classification = {
        'onestate': 'polynomial',
        'lorenz': 'polynomial',
        'twophase': 'phaseosc',
        'twostate1': 'polynomial',
        'twostate2': 'polynomial'
    }

    igramtype = [i for i in system_classification if sys_name in system_classification[i]]
    igram = grammar_classification[igramtype[0]]

    return [system_expressions[sys_name], system_params[sys_name], system_bounds[sys_name],
            system_symbols[sys_name], obs, obstxt, igram, igramtype]

