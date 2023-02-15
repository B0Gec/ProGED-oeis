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
        # print(data.shape)
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
                 sample_size = 2,
                 estimation_settings = {'simulate_separately': True,
                     'time_index': 0,
                     'target_variable_index': -1,
                     'objective_settings': {
                         'simulate_separately': True,
                         # 'rtol': 10**(-4),
                     },
                     # 'verbosity': 7,
                 },
                 verbosity = 1)
    ED.generate_models()
    print('models', ED.models)
    ED.fit_models(
        # estimation_settings={'simulate_separately': True, },
    )
    # ED.fit_models(estimation_settings={'verbosity': 2})
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1)
    args = parser.parse_args()
    test_equation_discoverer_ODE(args.dim)