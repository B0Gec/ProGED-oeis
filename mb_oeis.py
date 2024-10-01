"""
file with domb (do moeller-buchberger) to replace increase_one in doone.py
"""

from typing import Union
import sys
import re
import math

import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# TAKELESSTERMS = True  # take less terms than 200, to make MB faster and maybe even more precise.

from sindy_oeis import preprocess, solution_vs_truth
from exact_ed import solution2str, check_eq_man, solution_reference, dataset, lib2degrees, lib2verbose
from mb_wrap import mb


# sys.path.append('../monomial-agnostic-vanishing-ideal')
# from mavi.vanishing_ideal import VanishingIdeal
# # from mavi.mavi_eed_utils import simpl_disp
# from mavi_simplify import round_expr, divide_expr, simpl_disp, anform
#

def increasing_mb(seq_id, csv, max_order, n_more_terms, n_of_terms):
    """
    Run a for loop of increasing order where I run Moeller-Buchberger algorithm on a given sequence.
    """

    seq = unnan(csv[seq_id])[:n_of_terms]
    print(seq)

    for order in range(0, max_order + 1):
        ideal, ref = one_mb(seq, order)

        def one_mstb(seq_id, csv, order, n_more_terms, library='n', n_of_terms=200) -> tuple:

            print(f'order: {order}, all generators: \n)')
        print(ideal)
        # booly = check_eq_man(eq, seq_id, csv, header=False, n_of_terms=10 ** 5, solution_ref=ref, library='n')
        # if bolly
        #     break

    return ideal, ref


def one_mb(seq, order, n_more_terms, library='n', n_of_terms=200) -> tuple:
    """
    A/one mb (Moeller-Buchberger) algorithm, i.e. one run of MB algorithm (for one given recursion order).
    Simple function to put inside increase_mb without checking for correctness, only displaying.

    Args:
        - seq_id
        - csv
        - order (int): order of the recursion.
        - n_of_terms (int): number of terms to take from the sequence.

    Returns: list of generators produced by MB algorithm.
    """

    # take max_order + n_more_terms terms from given sequence terms
    # if TAKELESSTERMS:
    seq = seq[:order + n_more_terms]
    print(seq)

    # seq = sindy_hidden[d_max_lib - 1]
    # # print(n_of_terms, seq)
    # seq = seq[:n_of_terms]
    # # seq, coeffs, truth = unpack_seq(seq_id, csv)

    # seq = [float(i) for i in seq]
    # # print(f"{int(seq[-1]):.4e}")
    # # 1/0

    print(order)

    # b, A, sol_ref = dataset(list(seq), d_max=d_max_lib, max_order=max_order, library='n')
    b, A, sol_ref = dataset(list(seq), d_max=1, max_order=order, library=library)

    # Ignore the constant 1 column (mavi auto. deals with constants)
    A, sol_ref = A[:, 1:], sol_ref[1:]
    data = np.concatenate((b, A), axis=1)

    # How we would like to print variables:
    sol_ref = solution_reference(library, d_max=1, order=order)
    print(sol_ref)

    # Bijection mapping: printing like a(n-1) vs. cocoa: a_n_1:
    sol_ref_inverse = {var.replace('(', '_').replace('-', '_').replace(')', ''): var
                       for var in sol_ref[2:]}
    vars_cocoa = ['a_n', 'n'] + list(sol_ref_inverse.keys())
    sol_ref_inverse['a_n'] = 'a(n)'  # to avoid a(n)_2 situation by first replacing a_n
    print(vars_cocoa)
    print(sol_ref_inverse)

    # print(mb(points=data.tolist(), execute_cmd=False, visual='djus'))
    # print(mb(points=data.tolist(), execute_cmd=True, var_names='oeis'))
    # print(mb(points=data.tolist(), execute_cmd=True, var_names=var_names))
    # print(mb(points=data.tolist(), execute_cmd=False, var_names=vars_cocoa))
    first_generator, ideal = mb(points=data.tolist(), execute_cmd=True, var_names=vars_cocoa)
    # change e.g. a_n_1 to a(n-1). (i.e. apply the inverse)

    for key in sol_ref_inverse:
        ideal = ideal.replace(key, sol_ref_inverse[key])
        first_generator = first_generator.replace(key, sol_ref_inverse[key])
    print(ideal)
    print(first_generator)
    return ideal, ['a(n)'] + sol_ref[1:]


# if run as standalone:
if __name__ == '__main__':

    # import matplotlib.pyplot as plt
    # from mavi_oeis import anform

    eq = """
        0.01⋅a(n) - 0.004⋅a(n - 3) - 0.02⋅a(n - 1) + 0.02 = 0
        """
    # print(anform(eq, 0.01))
    # 1/0

    seq_id = 'A000045'
    seq_id = 'A000079'
    csv_filename = 'cores_test.csv'
    N_OF_TERMS_LOAD = 10 ** 5
    # N_OF_TERMS_LOAD = 20
    # N_OF_TERMS_LOAD = 10
    # N_OF_TERMS_LOAD = 2
    csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])[:N_OF_TERMS_LOAD]

    from IPython.display import display, display_latex
    from exact_ed import unnan

    seq, coeffs, truth = (unnan(csv[seq_id]), None, None)
    print(seq)
    # 1/0

    # # seq = sindy_hidden[d_max_lib - 1]
    # # print(n_of_terms, seq)
    # seq = seq[:N_OF_TERMS_LOAD]
    # print(seq)
    # # 1/0


    eqs = one_mb(seq_id, csv, verbosity=0, d_max_lib=1,
                d_max_mavi = 2, order = 1, ground_truth = False,
                n_of_terms = N_OF_TERMS_LOAD, library = 'n',
                start_order = None, init = None, sindy_hidden = [seq])

    1/0
    print()

    seq = seq[:N_OF_TERMS_LOAD]
    # seq, coeffs, truth = unpack_seq(seq_id, csv)

    seq = [float(i) for i in seq]
    # print(f"{int(seq[-1]):.4e}")
    # 1/0
    print(seq)

    d_max_mavi = 3
    max_order = 1
    print_digits = 2
    print_epsilon = 1e-01
    divisor = 1
    divisor = 0.45
    # b, A, sol_ref = dataset(list(seq), d_max=d_max_lib, max_order=max_order, library='n')
    b, A, sol_ref = dataset(list(seq), d_max=1, max_order=max_order, library='n')
    # ignore the constant 1 column (mavi auto. deals with constants)
    A, sol_ref = A[:, 1:], sol_ref[1:]
    data = np.concatenate((b, A), axis=1)
    data = data.astype('float')

    vi = VanishingIdeal()
    vi.fit(data, 0.01, method="grad", max_degree=d_max_mavi)
    vi.plot(data, target="vanishing", splitshow=True)
    # plt.show()


    data_symb_list = ['a(n)'] + sol_ref
    X_symb = np.array([sp.symbols(', '.join(data_symb_list))])

    G = vi.evaluate(X_symb, target='vanishing')  # (1, 6) array
    G = np.ravel(G)
    # for i, g in enumerate(G):
    #     g = sp.expand(g)
    #     print(g)
    # print(simpl_disp(g, verbosity=0, num_digits=2, epsilon=1e-10)[0])
    # display(mu.simpl_disp(g, verbosity=0, num_digits=2, epsilon=1e-10)[0])
    eqs = [sp.pretty(simpl_disp(sp.expand(g), verbosity=0, num_digits=print_digits, epsilon=print_epsilon)[0],
                     num_columns=400) for i, g in enumerate(G)]

    chosen = 0
    # print('eqs0 non divided:\n', eqs[chosen], '\n'*4)
    # for i in eqs:
    #     print(i)
    # print('eqs0 non divided:\'', eqs, '\n'*4)
    # divisor = 0.58
    # divisor = 1
    # # divisor = 0.24
    # divisor = -0.45
    # divisor = 0.09
    # divisor = 0.09
    eqs_div = [sp.pretty(divide_expr(simpl_disp(sp.expand(g), verbosity=0, num_digits=print_digits, epsilon=print_epsilon)[0],
                                     print_digits, divisor=divisor), num_columns=400) for i, g in enumerate(G)]

    # print(eqs)
    eqs = eqs_div
    # 1/0
    # print(['a(n)' in i for i in eqs])
    # print('\nall eqs:\n')
    # for i in eqs:
    #     print(i)
    # print('\n end of all eqs:\n')
    # readable_eqs = [anform(eq, rounding=print_digits) for eq in eqs[0:]]
    if len(eqs) > 0:
        readable_eq = anform(eqs[chosen], rounding=print_digits) if len(eqs) > 0 else 'No eq'

        # print(eqs)
        # print('eqs0 divided:\n', eqs[chosen])
        print('\nan form (linear display):\n', readable_eq, '\n'*4)
        1/0
    else:
        print('no eqs:\n', eqs)

    for n, e in enumerate(eqs):
        print(f'poly #{n}')
        print(e)
        # print(sp.latex(i))
        # print('polatex')
        # print(sp.pretty(e, num_columns=300))
        # sp.pprint(i)
        print()

