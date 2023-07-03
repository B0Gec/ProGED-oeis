from typing import Union
from itertools import product

import pandas as pd
import sympy as sp
import numpy as np
import pysindy as ps

from exact_ed import grid_sympy, dataset, unpack_seq, truth2coeffs, solution_vs_truth, instant_solution_vs_truth, solution2str, check_eq_man

"""
todo:
 - n = 30 .... except if      abs(i) > 10**16 for any i in seq. Then half-half strategy n -> order=n/2, n/2 slices.
 
 
 1 2 3 4 .... 30

 1 2 3 4 5
 2 3 4 5 6
 3 4 5 6 7
 ...
 26 27 28 29 30
 
 
 Sindy fails:
   - too sparse (0 vs 1*a(n-1), a(n-1) vs a(n-1)+a(n-2), ... )
   - uncorrect param estimated even with correct order set in dataset

 To try:
   - lower treshold from 0.1 to 0.05?
   
   
   
   grid:
   
    max_order x seq_len
        1..20 x 200/10 = 20
       # 5, ...  
       # seq len: len ... 200 
       20 
       [1, len]/20 =     
       [i for i in range(1, 20+1)] x [i for i in range(4, len, (len-4)/20)]
       l= 30;s=[round(4+i*(l-4)/20) for i in range(20)];len(s),s
       l= 10;s=list(set([round(4+i*(l-4)/20) for i in range(20)]));len(s),s

   
"""


def heuristic(terms_avail):
    """Calculate/decide on max_order given the number of available terms (of size < 10**16)."""
    return round(terms_avail/2)

def preprocess(seq):
    """Filter in only small sized terms of the sequence."""

    biggie = list(map(lambda term: abs(term) >= 10**16, seq))
    fail = False
    if True in biggie:
        pos = biggie.index(True)
        if pos <= 0:
            print('Only huge terms in the sequence!!!')
            fail = True
            return [], fail
        else:
            if pos <= 2:
                print('Only huge terms in the sequence!!!')
                fail = True
            return seq[:pos], fail
    else:
        return seq, fail

def sindy(seq: Union[list, sp.Matrix], max_order: int, seq_len: int, threshold: float = 0.1):
    """Perform SINDy."""

    # Generate training data
    # seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    #        1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    #        121393, 196418, 317811, 514229, 832040, 1346269,
    #        2178309, 3524578, 5702887, 9227465, 14930352, 24157817,
    #        39088169, 63245986, 102334155]

    # print(len(ongrid))
    seq = [int(i) for i in seq][:seq_len]
    # print(type(seq), type(seq[-1]))
    # seq = seq[:90]
    # seq = seq[:70]
    # seq = seq[:30]
    # print(f"{int(seq[-1]):.4e}")
    # 1/0

    b, A = dataset(seq, max_order, linear=True)
    # b, A = dataset(seq, 19, linear=True)
    # b, A = dataset(sp.Matrix(seq), 2, linear=True)
    # 2-7, 9-14, 16-19 finds, 8,15 not
    b, A = np.array(b, dtype=int), np.array(A, dtype=int)
    # print(b, A)
    # 1/0
    # data = grid_sympy(sp.Matrix(seq), max_order)
    # data = sp.Matrix.hstack(b, A)
    data = np.hstack((b, A))

    # print(data.shape, type(data))
    head = data[:6, :6]
    # for i in range(head.rows):
    # for i in range(head.shape[0]):
    #     print(data[i, :6])

    # for i in range(data[:6, :].rows):
    #     print(data[:i, :])
    #
    # print(data)


    # poly_order = 8
    poly_order = 1
    # threshold = 0.1

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
        feature_names=[f"a(n-{i+1})" for i in range(max_order-1)],
        discrete_time=True,
    )


    # # model.fit(x_train, t=dt)
    # model.fit(x_train, t=dt, x_dot=dot_x)
    model.fit(A, x_dot=b)
    model.fit(A, x_dot=b, ensemble=True)
    model.fit(A, x_dot=b, library_ensemble=True)
    # model.print()
    model.coefficients()
    print(model.coefficients(), 'model.coefficients()')
    print(type(model.coefficients()), 'type of model.coefficients()')
    print(np.mean(model.coef_list, axis=0), 'model.coefficients()')
    # print(type(model.coef_list), 'type of model.coefficients()')
    1/0
    # print(model.coefficients())
    x = sp.Matrix([round(i) for i in model.coefficients()[0][1:]])
    x = sp.Matrix.vstack(sp.Matrix([0]), x)

    # print(x)
    return x



def sindy_grid_search(seq, coeffs, truth, max_order: int, stepsize: int, evals: int):
    """Performs sindy on grid of parameters until the correct sindy equation is found according to the ground truth.
    Outputs:
        - list of tuples of coeffs, truth values, chosen parameter values, for each try ...
    """

    # [orders .... 20]
    # [n_of_terms   200]

    # success = []
    # for order in range(max_order):
    #
    #     # seq, coeffs, truth = unpack_seq(seq_id, csv)
    #     x = sindy(list(seq), order)
    #     is_reconst = solution_vs_truth(x, coeffs)
    #     is_check_verbose = check_eq_man(x, seq_id, csv, n_of_terms=10 ** 5)
    #     is_check = is_check_verbose[0]
    #     success += (x, is_reconst, is_check, order)
    #
    #     if is_reconst:
    #         return x,
    #
    return


def one_results(seq, seq_id, csv, coeffs, max_order: int, seq_len: int):

    # print(max_order, seq_len)
    x = sindy(seq, max_order, seq_len)
    is_reconst = solution_vs_truth(x, coeffs)
    is_check_verbose = check_eq_man(x, seq_id, csv, n_of_terms=10 ** 5)
    is_check = is_check_verbose[0]
    summary = x, is_reconst, is_check, max_order
    # print()
    # print(summary)
    # print('one result !!!!!!!')
    # print(max_order, seq_len)
    return summary


def sindy_grid_order(seq, seq_id, csv, coeffs, max_order: int, seq_len: int):
    # weird lazy error: !!!!!
    # for i in range(2, 5):
    #     print(one_results(seq, seq_id, csv, coeffs, i), [i for i in range(3, 8)])

    return map(lambda order: one_results(seq, seq_id, csv, coeffs, order, seq_len), [i for i in range(1, max_order+1)])

def sindy_grid(seq, seq_id, csv, coeffs, max_order: int, seq_len: int, grid_order: int = 20, grid_len: int = 20):
    # weird lazy error: !!!!!
    # for i in range(2, 5):
    #     print(one_results(seq, seq_id, csv, coeffs, i), [i for i in range(3, 8)])

    seq_len = min(len(seq), seq_len)


    def equidist(start, end, n_of_pts):
        """Returns list of n_of_pts equidistant points between start and end.

        E.g. equidist(1, 10, 3) = [1, 5, 10]
        """
        return list(set([round(start + i * (end - start) / n_of_pts) for i in range(n_of_pts)]));

    # todo grid 20x20x20 (10h for experiment) where 20 for different values of (sindy's) threshold.
    subopt_grid = list(product(equidist(1, max_order, grid_order), equidist(4, seq_len, grid_len)))  # i.e.
    grid = [pair for pair in subopt_grid if (pair[1]-pair[0]) > 4]  # Avoids too short sequences vis-a-vis order.
    # grid = grid[:6]

    printout = str(grid)
    # print(grid)

    ongrid = list(map((lambda order_leng: (order_leng[0], order_leng[1], one_results(seq, seq_id, csv, coeffs, order_leng[0], order_leng[1])[:3])), grid))
    printable = [(order, leng, oneres[1], oneres[2]) for order, leng, oneres in ongrid]
    printout += '\n'
    for i in range(round(len(printable)/5) + 1):
        printout += ', ' + "".join([f"{str(i): >22}, " for i in printable[5*i:5*(i+1)]]) + '\n'
        # printout += str([f"{str(i): >20}" for i in printable[5*i:5*(i+1)]])
    # printout += ']\n'

    oeis = [case[2][0] for case in ongrid if case[2][1:3] == (True, True)]
    manually = [case[2][0] for case in ongrid if case[2][1:3] == (False, True)]
    # bug = [case[2][0] for case in ongrid if case[2][1:3] == (True, False)]
    fail = [case[2][0] for case in ongrid if case[2][1:3] == (False, False)]
    printout += "\n"
    printout += f"number of all configurations: {len(grid)}\n"
    printout += f"number of fully (true, true) successful configurations: {len(oeis)}\n"
    printout += f"number of partially only (false, true) successful configurations: {len(manually)}\n"
    printout += f"number of total fail (false, false) configurations: {len(fail)}\n"
    # allxs = [case[2][0] for case in ongrid]
    # sp.Matrix([round(i) for i in list(sum(ll, sp.Matrix([0, 0, 0])) / 3)])

    # print(xs)
    if oeis != []:
        x = oeis[0]
    elif manually != []:
        x = manually[0]
    else:
        x = []
        # x = sp.Matrix([0, 0, 0, 0])
    # print(x)

    xs = [case[2][0] for case in ongrid]
    # print([len(x) for x in xs])
    max_len = max([len(x) for x in xs])
    xs = [sp.Matrix.vstack(x, sp.zeros(max_len-len(x), 1)) for x in xs]
    # x_avg = sp.Matrix([round(i) for i in list(sum(xs, sp.Matrix([0, 0, 0])) / 3)])
    x_avg = sp.Matrix([round(i) for i in list(sum(xs, sp.zeros(max_len, 1)) / len(xs))])
    # print('xs', xs)
    # print('x_avg', x_avg)


    # [i for i in range(1, 20 + 1)]
    # x[i
    # for i in range(4, len, (len - 4) / 20)]
    # l = 30;
    # s = [round(4 + i * (l - 4) / 20) for i in range(20)];
    # len(s), s
    # l = 10;
    # s = list(set([round(4 + i * (l - 4) / 20) for i in range(20)]));
    # len(s), s

    # return map(lambda order: one_results(seq, seq_id, csv, coeffs, order, seq_len), [i for i in range(1, max_order+1)])
    return x, printout, x_avg

# todo + sindy ensemble (with different thresholds) x 2 versions (ensamble + library_ensemble)


if __file__ == '__main__':
    csv_filename = 'linear_database_full.csv'

    # if CORELIST:
    #     # from core_nice_nomore import cores
    #     csv_filename = 'cores.csv'

    # # print(os.getcwd())
    # if os.getcwd()[-11:] == 'ProGED_oeis':
    #     # csv_filename = 'ProGED_oeis/examples/oeis/' + csv_filename
    #     # print(os.getcwd())
    #     pass
    # # except ValueError as error:
    # #     print(error.__repr__()[:1000], type(error))

    seq_id = 'A000045'
    # csv = pd.read_csv(csv_filename, low_memory=False, nrows=0)
    csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])
    # print(csv)

    print(unpack_seq(seq_id, csv))
    print(unpack_seq(seq_id, csv)[1])
    print(solution2str(sp.Matrix([1, 0, 1, 0, 0, 0, 1])))
    print(solution2str([]))
    # print(solution_vs_truth([]))
    print(instant_solution_vs_truth(sp.Matrix([1, 0, 1, 0, 0, 0, 1]), seq_id, csv))
    print(instant_solution_vs_truth(sp.Matrix([0, 1, 1, 0, 0, 0, 0]), seq_id, csv))
    print(instant_solution_vs_truth(sp.Matrix([0, 1, 1, 0, 0, 0, 0]), seq_id, csv))
    print(solution_vs_truth(sp.Matrix([0, 1, 1, 0, 0, 0, 0]), sp.Matrix([1, 1])))

    seq, coeffs, truth = unpack_seq(seq_id, csv)



