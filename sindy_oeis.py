import warnings
warnings.simplefilter("ignore")

from typing import Union
from itertools import product

import pandas as pd
import sympy as sp
import numpy as np
import pysindy as ps

warnings.filterwarnings('ignore', module='pysindy')
from exact_ed import grid_sympy, dataset, unpack_seq, truth2coeffs, solution_vs_truth, instant_solution_vs_truth, solution2str, check_eq_man, lib2degrees

import warnings
warnings.filterwarnings("ignore")

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

def preprocess(seq: sp.Matrix, library: str) -> tuple[sp.Matrix, bool]:
    """Filter in only small sized terms of the sequence."""

    n_degree, degree = lib2degrees(library)
    # biggie = list(map(lambda term: abs(term) >= 10**16, seq))
    biggie = list(map(lambda term: abs(term**degree) >= 10**16, seq))
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

def sindy(seq: Union[list, sp.Matrix], max_order: int, seq_len: int, threshold: float = 0.1,
          ensemble: bool = False, library_ensemble: bool = False, library: str = 'lin'):
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

    b, A = dataset(seq, max_order, library=library)
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

    # max_degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('cub', 'ncub') else 'Unknown Library!!'
    _, max_degree = lib2degrees(library)

    # poly_order = 8
    poly_order = 1
    poly_order = max_degree
    # threshold = 0.1

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
        feature_names=[f"a(n-{i+1})" for i in range(max_order-1)],
        discrete_time=True,
    )


    # # model.fit(x_train, t=dt)
    # model.fit(x_train, t=dt, x_dot=dot_x)
    # model.fit(A, x_dot=b)
    # model.fit(A, x_dot=b, ensemble=True)
    # model.fit(A, x_dot=b, library_ensemble=True)
    model.fit(A, x_dot=b, ensemble=ensemble, library_ensemble=library_ensemble)
    # model.print()
    model.coefficients()
    # print(model.coefficients())
    x = sp.Matrix([round(i) for i in model.coefficients()[0][1:]])
    # x = sp.Matrix.vstack(sp.Matrix([0]), x)

    # model.print()

    print(x)
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


def one_results(seq, seq_id, csv, coeffs, max_order: int, seq_len: int,
                threshold: float, library: str, ensemble: bool, library_ensemble: bool):

    # print(max_order, seq_len)
    x = sindy(seq, max_order, seq_len, threshold, ensemble, library_ensemble, library)
    is_reconst = solution_vs_truth(x, coeffs) if coeffs is not None else ' - NaN - '
    is_check_verbose = check_eq_man(x, seq_id, csv, n_of_terms=10 ** 5, library=library)
    print('oner', x, library, is_check_verbose)
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


def create_grid(seq, seq_id, csv, max_order: int, seq_len: int,
                ths_bounds: tuple = (0, 0.9), ensemble_grid: tuple = (True, False, False),
                order_pts: int = 20, len_pts: int = 20, threshold_pts: int = 20):

    seq_len = min(len(seq), seq_len)

    def equidist(start, end, n_of_pts):
        """Returns list of n_of_pts equidistant points between start and end.

        E.g. equidist(1, 10, 3) = [1, 5, 10]
        """
        if n_of_pts == 1:
            return [round(end)]
        else:
            return list(set([round(start + i * (end - start) / n_of_pts) for i in range(n_of_pts)]))

    # todo grid 20x20x20 (10h for experiment) where 20 for different values of (sindy's) threshold.

    order_grid = equidist(1, max_order, order_pts)
    # print('ord', order_grid)
    terms_grid = equidist(4, seq_len, len_pts)
    # print(terms_grid)
    threshold_grid = np.linspace(ths_bounds[0], ths_bounds[1], threshold_pts)
    ensemble_grid = [i for i in range(len(ensemble_grid)) if ensemble_grid[i]]
    # print(ensemble_grid)

    # subopt_grid = list(product(equidist(1, max_order, grid_order), equidist(4, seq_len, grid_len)))  # i.e.
    subopt_grid = list(product(order_grid, terms_grid, threshold_grid, ensemble_grid))  # i.e.
    # print(subopt_grid)
    grid = [pair for pair in subopt_grid if (pair[1]-pair[0]) > 4]  # Avoids too short sequences vis-a-vis order.
    return grid


def sindy_grid(seq, seq_id, csv, coeffs,
               max_order: int, seq_len: int, library: str):
               # ths_bounds: tuple = (0, 0.9), ensemble_grid: tuple = (True, False, False),
               # order_pts: int = 20, len_pts: int = 20, threshold_pts: int = 20, grids: dict = None):
               # grids={'max_order': 20, 'seq_len': 70, 'threshold': (0, 0.9),
               #        'ensemble': (True, False, False), 'library_ensemble': (True, False, False)}):
    """Performs sindy on grid of parameters until the correct sindy equation is found according to the ground truth.
    Otherwise outputs more complex equation or nothing.

    Args:
        ths_bounds: (min, max) for threshold
        ensemble_grid: (default, ensemble, library_ensemble)
        order_pts: number of points for order grid
        len_pts: number of points for sequence length grid
        threshold_pts: number of points for threshold grid
    """
    # weird lazy error: !!!!!
    # for i in range(2, 5):
    #     print(one_results(seq, seq_id, csv, coeffs, i), [i for i in range(3, 8)])

    # todo: default seq_len depending on huge values of terms in seq!!!!.

    #plan:
    # 3 grids:
    # 1) order x len x ths x ens:
    #            20*20*18 + 20*18*2 = 7920
    #            grid (ord=20, len=20, ths=18, ens=0)
    #            grid (ord=20,  ths=18, ens=1)
    #            grid (ord=20,  ths=18, ens=2)

    grid1 = create_grid(seq, seq_id, csv, max_order, seq_len,
                       ths_bounds = (0, 0.9), ensemble_grid=(True, False, False),
                       # order_pts=20, len_pts=20, threshold_pts=18)
                       order_pts = 20, len_pts = 10, threshold_pts = 10)
    grid2 = create_grid(seq, seq_id, csv, max_order, seq_len,
                       ths_bounds=(0, 0.9), ensemble_grid=(False, True, True),
                       order_pts=20, len_pts=1, threshold_pts=10)
    # small_grid = create_grid(seq, seq_id, csv, max_order, seq_len,
    #                    ths_bounds = (0, 0.9), ensemble_grid=(True, False, False),
    #                    # order_pts=20, len_pts=20, threshold_pts=18)
    #                    order_pts = 2, len_pts = 2, threshold_pts = 3)

    small_grid = create_grid(seq, seq_id, csv, max_order, seq_len, ths_bounds = (0, 0.9),
                             ensemble_grid=(True, False, False),
                           order_pts=3, len_pts=2, threshold_pts=3)
    # small_grid = create_grid(seq, seq_id, csv, coeffs, 4, seq_len,
    #                         ths_bounds = (0.1, 0.2), ensemble_grid=(False, False, True),
    #                         order_pts=3, len_pts=3, threshold_pts=3)
    grid = grid1 + grid2
    # grid = grid1
    # grid = grid2
    grid = small_grid


    # 2x2x3 = 12 + 2x1x3x2 = 12 ... = 24

    # if grid is none
    # grid = []
    # for grid in grids:
    #     grid += create_grid(**grid)

    # grid = grid[:3000]
    # grid = grid[4000:4200]
    five = []
    for i in range(len(grid)):
        if i%5 == 0:
            # print(five)
            five = []
        five += [grid[i]]
    # print(five)
    # print(grid)
    # print(len(grid))
    # print(len(grid), 'of', order_pts * len_pts * threshold_pts + len(ensemble_grid))
    # 1/0

    # printout = str(grid)
    printout = ''
    # print('here grid', grid)
    def ens2dict(ensemble):
        if ensemble == 0:
            return {'ensemble': False, 'library_ensemble': False, }
        elif ensemble == 1:
            return {'ensemble': True, 'library_ensemble': False, }
        else:
            return {'ensemble': False, 'library_ensemble': True, }

    ongrid = list(map(
        (lambda i: i + (one_results(seq, seq_id, csv, coeffs, i[0], i[1], i[2], library, **ens2dict(i[3]))[:3], )), grid))
    print(ongrid)
    printable = [(order, leng, thrs, ens, oneres[1], oneres[2]) for order, leng, thrs, ens, oneres in ongrid]
    printout += '\n'
    for i in range(round(len(printable)/5) + 1):
        printout += ', ' + "".join([f"{str(i): >22}, " for i in printable[5*i:5*(i+1)]]) + '\n'
        # printout += str([f"{str(i): >20}" for i in printable[5*i:5*(i+1)]])
    # printout += ']\n'

    if len(grid) == 0:
        printout += f"\nNo configurations to test!\n"
        oeis, manually, fail = [], [], []
    else:
        print(grid[0])
        print(ongrid[0])
        # for i in grid:
        #     print(i)
        idx = len(grid[0])
        print(idx)
        print(ongrid[0][idx])
        # print(grid[0][idx][1:3])
        # 1/0
        oeis = [case[idx][0] for case in ongrid if case[idx][1:3] == (True, True)]
        # manually = [case[idx][0] for case in ongrid if case[idx][1:3] == (False, True)]
        manually = [case[idx][0] for case in ongrid if case[idx][2] == True and (not case[idx][0] in oeis)]
        # bug = [case[idx][0] for case in ongrid if case[idx][1:3] == (True, False)]
        # fail = [case[idx][0] for case in ongrid if case[idx][1:3] == (False, False)]
        fail = [case[idx][0] for case in ongrid if case[idx][2] == False]
        # print(idx, oeis, manually)
        # 1/0

    printout += "\n"
    printout += f"number of all configurations: {len(grid)}\n"
    printout += f"number of fully (true, true) successful configurations: {len(oeis)}\n"
    printout += f"number of partially only (false, true) successful configurations: {len(manually)}\n"
    printout += f"number of total fail (false, false) configurations: {len(fail)}\n"
    # allxs = [case[2][0] for case in ongrid]
    # sp.Matrix([round(i) for i in list(sum(ll, sp.Matrix([0, 0, 0])) / 3)])

    # print(xs)
    def selection_criteria(l: list) -> sp.Matrix:
        """Choose solution of minimal "order", should I say complexity, i.e. lowest number of terms."""

        if l == []:
            return l
        else:
            # print('l', l)

            # calculate complexity of given solution x (by counting number of non-zero terms):
            def cmplx(x): return len([None for term in x if term != 0])
            # complexities = [len([None for term in x if term != 0]) for x in l]
            # min_compx = min(complexities)
            # winners are those with minimal complexity:
            winners = [x for x in l if cmplx(x) == min(map(cmplx, l))]
            # if len(winners) > 1:
            # winners = [x for x in winners if x[0] == min(map(lambda x: x[0], winners))]
            # print('winners', winners)
            # locations
            locationss = [[n for n, i in enumerate(x) if i != 0] for x in winners]
            # print('locationss', locationss)
            # subwinners = [winners[n] for n, locs in enumerate(locationss) if sum(locs) == min(map(sum, locationss))]
            subwinners = [(winners[n], locs) for n, locs in enumerate(locationss) if sum(locs) == min(map(sum, locationss))]
            # map(lambda x: x[-1], [loc for i in locationss)
            # print('subwinners', subwinners)
            real_winner = subwinners[[locs[-1] for _, locs in subwinners].index(min([locs[-1] for _, locs in subwinners]))][0]


            # print('real_winner', real_winner)
            return real_winner

    x = selection_criteria(oeis if oeis != [] else manually if manually != [] else [])
    # print(x)
    # print(selection_criteria(oeis+manually+fail))


    # if oeis != []:
    #     x = oeis[0]
    # elif manually != []:
    #     x = manually[0]
    # else:
    #     x = []
    #     # x = sp.Matrix([0, 0, 0, 0])
    # print(x)

    # print(ongrid[idx])
    xs = [case[idx][0] for case in ongrid]
    # print([len(x) for x in xs])
    max_len = max([len(x) for x in xs]) if xs != [] else 0
    # print(max_len)
    xs = [sp.Matrix.vstack(x, sp.zeros(max_len-len(x), 1)) for x in xs]
    # print(xs)
    # x_avg = sp.Matrix([round(i) for i in list(sum(xs, sp.Matrix([0, 0, 0])) / 3)])
    x_avg = sp.Matrix([round(i) for i in list(sum(xs, sp.zeros(max_len, 1)) / len(xs))]) if xs != [] else []
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


if __name__ == '__main__':
    csv_filename = 'linear_database_full.csv'

    # import warnings

    warnings.filterwarnings('ignore', module='pysindy')
    # warnings.simplefilter("ignore")

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

    warnings.filterwarnings('ignore', module='pysindy')
    # warnings.simplefilter("ignore")
    x = sindy(seq, max_order = 2, seq_len = 10, threshold=21.9)
    print(x)

