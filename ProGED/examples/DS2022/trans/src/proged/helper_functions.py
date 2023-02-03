import itertools
import numpy as np


# get parts of expression or equations (terms), that are either in or not in the system examined. This
# is used to find the right structures after grammar generates too many models.
def get_fit_settings(obs):
    objective_settings = {
        "atol": 10 ** (-6),
        "rtol": 10 ** (-4),
        "use_jacobian": False,
        "teacher_forcing": False,
        "simulate_separately": False,
        "persistent_homology": False,
        "persistent_homology_size": 200,
        "persistent_homology_weight": 0.5,
    }

    optimizer_settings = {
        "lower_upper_bounds": (-10, 10),
        "default_error": 10 ** 9,
        "strategy": 'best1bin', # best1bin
        "f": 0.5, #(0.5, 1.0), # 0.8
        "cr": 0.5,
        "max_iter": 20,
        "pop_size": 4,
        "atol": 0.001,
        "tol": 0.001,
        "pymoo_min_f": 10**(-5)
    }

    estimation_settings = {
        "target_variable_index": None,
        "time_index": 0,
        "max_constants": 10,
        "optimizer": 'DE_pymoo',  # DE_pymoo, DE_manual, differential_evolution
        "observed":  obs,
        "optimizer_settings": optimizer_settings,
        "objective_settings": objective_settings,
        "default_error": 10 ** 9,
        "timeout": np.inf,
        "verbosity": 0,
    }

    return estimation_settings


def get_opt_params(idx):
    fs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    crs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_combinations = list(itertools.product(fs, crs))
    return all_combinations[idx][0], all_combinations[idx][1]