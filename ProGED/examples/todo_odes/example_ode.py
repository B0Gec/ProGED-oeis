# -*- coding: utf-8 -*-
import numpy as np
from ProGED.equation_discoverer import EqDisco
import argparse

def test_equation_discoverer_ODE(dim: int):
    ts = np.linspace(1, 5, 128)
    xs = np.exp(ts)
    ys = np.exp(ts)

    if dim==1:
        # ERROR
        data = np.hstack((ts.reshape(-1, 1), xs.reshape(-1, 1)))
        var_names = ["t", "x"]
    elif dim==2:
        # seems to work
        data = np.hstack((ts.reshape(-1, 1), xs.reshape(-1, 1), ys.reshape(-1, 1)))
        var_names = ["t", "x", "y"]

    np.random.seed(0)
    ED = EqDisco(data = data,
                 task = None,
                 task_type = "differential",
                 time_index = 0,
                 target_variable_index = -1,
                 variable_names=var_names,
                 sample_size = 4,
                 estimation_settings = {
                     # 'time_index': 0,
                     # 'target_variable_index': -1,
                     # 'verbosity': 3,
                     'objective_settings': {
                         'simulate_separately': True,
                     },
                 },
                 verbosity = 2)
    ED.generate_models()
    ED.fit_models()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1)
    args = parser.parse_args()
    test_equation_discoverer_ODE(args.dim)