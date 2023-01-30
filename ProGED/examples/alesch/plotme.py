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

def pxy(sample_size):
    var_names = ["x", "y", ]
    # var_names = ["x", "y", 'w']
    grammar_template_name = "polynomial"
    # grammar_template_name = "polynomial2"
    # grammar_template_name = "universal"
    lower_upper_bounds = (-5, 5)  # direct
    p_T = [0.4, 0.6]  # default settings, does nothing
    # p_T = [0.9, 0.1]
    # p_R = [0.8, 0.2]
    p_R = [0.6, 0.4]  # default settings, does nothing
    functions = []
    p_F = [0.333, 0.333, 0.333]  # before exact
    functions = ["'sqrt'", "'exp'", "'log'",]  # before exact

    generator_settings = {
        "functions": functions,
        "p_T": p_T, "p_R": p_R,
        # "p_R": p_R,
        "p_F": p_F,
    }
    timeout = np.inf

    # print('variable_names', variable_names)

    np.random.seed(0)
    ED = EqDisco(
        # data=cso,
        data=cs,
        task=None,
        target_variable_index=-1,
        variable_names=var_names,
        # sample_size = 10,  # for direct fib
        # sample_size=124,
        sample_size=sample_size,
        # verbosity=0,
        verbosity = 0,
        generator="grammar",
        generator_template_name=grammar_template_name,
        # generator_settings=generator_settings,
        estimation_settings={
            "objective_settings": {"focus": (1, 0.5),},
            # "verbosity": 3,
            # 'target_variable_index': -1,
            # "verbosity": 1,
            # "verbosity": 0,
            "task_type": 'algebraic',
            "lower_upper_bounds": lower_upper_bounds,
            "optimizer": 'differential_evolution',
            # "optimizer": 'hyperopt',
            # "timeout": timeout,
            # "timeout_privilege": 30,
        }
    )


    print(f"=>> Grammar used: \n{ED.generator}\n")
    # # for i in [4,7,9]:
    # # for i in [seed, ]:
    # # np.random.seed(i)
    # # print('seed:', i)
    ED.generate_models()
    print(ED.models)
    ED.fit_models()
    # print(ED.models.retrieve_best_models(10000))
    # print(ED.models)
    print(ED.models.retrieve_best_models(-1))
    # return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ss", type=int, default=sample_size)  # = sample_size
    args = parser.parse_args()
    parser.add_argument("--sample_size", type=int, default=args.ss)
    args = parser.parse_args()
    pxy(args.sample_size)

#
# 0.274 - 1.995/(32.78 - 15.22 sqrt(-x_0 + 6.647-7exp(14.569*x_0) + 0.775))
# (0.03x0 - 0.063)(−11.104x0+(0.931x0−0.05)tan(3.22x0−1.74)+0.46)
