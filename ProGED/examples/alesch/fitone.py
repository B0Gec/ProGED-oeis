import pandas as pd
import numpy as np
import argparse

csv = pd.read_csv('ales.csv')
cs = np.array(csv)
# cso = np.hstack((cs, np.zeros((cs.shape[0], 1))))
# import ProGED as pg
# import ProGED

from ProGED.equation_discoverer import EqDisco


sample_size = 10 ** 3
sample_size = 100
sample_size = 10
# sample_size = 1
# sample_size = 10100

from ProGED import ModelBox
from ProGED.parameter_estimation import fit_models
models = ModelBox()
exprs = ["C*x",
         "C*x + C",
         "C*x + C*exp(C*x)",
         ]
symbols = {"x": ['x'], "const": "c", "start": "S"}
for expr in exprs:
    models.add_model(expr, symbols)
# print(models.add_model(expr1_str, symbols))
# models.add_model(expr2_str, symbols)
print(models)
# print(models.add_model(expr2_str))

# ED =
# 0.5 - 0.5*exp(-1.6*x)
# ED.generate_models()
# print(ED.models)

modelb = fit_models(models, data=cs, task_type="algebraic", pool_map=map,
                    estimation_settings={ 'target_variable_index':-1,
                                          # "lower_upper_bounds": (-5, 5),
                                          'optimizer_settings': {
                                            "lower_upper_bounds": (-3, 3),
                                            "max_iter": 18000,
                                            "pop_size": 150,
                                            }
                                            })


# fit_models()
# print(ED.models.retrieve_best_models(10000))
# print(ED.models)
print(modelb.retrieve_best_models(-1))
# return
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--ss", type=int, default=sample_size)  # = sample_size
#     args = parser.parse_args()
#     parser.add_argument("--sample_size", type=int, default=args.ss)
#     args = parser.parse_args()
#     pxy(args.sample_size)
#
#
# 0.274 - 1.995/(32.78 - 15.22 sqrt(-x_0 + 6.647-7exp(14.569*x_0) + 0.775))
# (0.03x0 - 0.063)(−11.104x0+(0.931x0−0.05)tan(3.22x0−1.74)+0.46)