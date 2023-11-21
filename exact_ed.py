"""Discover equation(s) on OEIS sequence to discover direct, recursive or even direct-recursive equation.
"""
import itertools
import os
import sympy as sp
import pandas as pd
import time
import math
# from typing import Union
from functools import reduce
from itertools import product


# if os.getcwd()[-11:] == 'ProGED_oeis':
#     from ProGED_oeis.ProGED.diophantine_solver import diophantine_solve
# else:
# from ProGED_oeis.diophantine_solver import diophantine_solve
from diophantine_solver import diophantine_solve

# print("IDEA: max ORDER for GRAMMAR = floor(DATASET ROWS (LEN(SEQ)))/2)-1")

def timer(now, text=f"\nScraping all (chosen) OEIS sequences"):
    before, now = now, time.perf_counter()
    consumed = now - before
    printout = f"{text}" \
          f" took:\n {round(consumed, 1)} seconds," \
          f" i.e. {round(consumed / 60, 2)} minutes" \
          f" or {round(consumed / 3600, 3)} hours."
    # print(printout)
    return now, printout



# def grid_numpy(seq_id: str, number_of_terms: int):
#     seq = np.array(sp.Matrix(list(csv[seq_id])[:number_of_terms]).T)[0]
#     n = len(seq)
#     # n = seq.shape[0]
#     indexes = np.fromfunction((lambda i,j: np.maximum(i-j,0)) , (n-1, n-1)).astype(int)
#     cut_zero = seq[indexes] * np.tri(n-1).astype(int)
#     data = np.hstack((np.array(seq)[1:].reshape(-1, 1), np.arange(1, n).reshape(-1, 1), cut_zero))
#     return data

# seq = sp.Matrix(csv[seq_id])
# def grid_sympy(seq: sp.MutableDenseMatrix, nof_eqs: int = None):  # seq.shape=(N, 1)

def poly_combinations(library: str, d_max: int, order: int):
    """Return all combinations of variables in library up to order.
    To avaid repetition in solution2str functiion.

    Example:
        library = 'n', d_max = 2, order = 1 -> ['n', 'a(n-1)', 'n*n', 'n*a(n-1)', 'a(n-1)*a(n-1)']
    """

    # n_degree, degree = lib2degrees(library)
    # basis = lib2stvars(library, order)
    basis = lib2stvars(library, order)
    # print('base', basis, library)

    combins = itertools.combinations_with_replacement
    # if library == 'n':
    #     basis, degree = ['n'], n_degree
    # print('lib', library, basis)
    combinations = sum([list(combins(basis, deg)) for deg in range(1, d_max+1)], [])
    # print('combinations inside poly_combinations()', combinations)
    return combinations

def solution_reference(library: str, d_max: int, order: int) -> list[str]:
    """Return reference solution for library and order."""
    # print((map(lambda comb: reduce((lambda i, s: i+'*'+s), comb[1:], comb[0]),  poly_combinations(library, order))))
    return ['1', ] + list(map(lambda comb: reduce((lambda i, s: i+'*'+s), comb[1:], comb[0]),  poly_combinations(library, d_max, order)))


multiply_eltw = lambda x,y: x.multiply_elementwise(y)
def comb2act(comb: tuple, dic: dict, multiply=multiply_eltw) -> sp.Matrix: return dic[comb[0]] if len(comb) == 1 else multiply(dic[comb[0]], comb2act(comb[1:], dic))


def grid_sympy(seq: sp.MutableDenseMatrix, d_max: int, max_order: int, library: str) -> tuple[sp.Matrix, list[str]]:  # seq.shape=(N, 1)
    # seq = seq if nof_eqs is None else seq[:nof_eqs]
    # seq = seq[:nof_eqs, :]
    # seq = seq[:shape[0]-1, :]
    # n = len(seq)

    n_degree, degree = lib2degrees(library)
    # print('lib, order, n_deg, degree:', library, max_order, n_degree, degree)
    # max_degree = 0 if library == 'n' else 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('cub', 'ncub') else 'Unknown Library!!'
    # n_cols = sp.Matrix.hstack(*[sp.Matrix([n**degree for n in range(1, seq.rows)]) for degree in range(1, n_degree+1)])
    # # n_col = sp.Matrix([n for n in range(1, seq.rows)])
    # # print('n_col', n_col.shape, n_col)
    # # ntriangles = triangle_grid(1)


    def triangle_grid(degree):
        return sp.Matrix(seq.rows-1, max_order,
                        (lambda i, j: (seq[max(i-j,0)]**degree)*(1 if i>=j else 0))
                        )

    basis = lib2stvars(library, max_order)
    # print(basis)

    n_col = sp.Matrix([i for i in range(1, seq.rows)]) if 'n' in basis else sp.Matrix([])
    # n_col = sp.Matrix([i for i in range(0, seq.rows-1)])
    # print('ncol, triangle grid(1)', n_col.__repr__(), triangle_grid(1).__repr__())

    ntriangle = sp.Matrix.hstack(n_col, triangle_grid(1))
    # print('ntriangle', ntriangle.shape, ntriangle.__repr__())
    triangle = {var: ntriangle[:, i] for i, var in enumerate(basis)}
    # print('triangle', triangle.keys())

    # combins = itertools.combinations_with_replacement
    # combinations = sum([list(combins(basis, deg)) for deg in range(1, degree+1)], [])
    combinations = poly_combinations(library, d_max, max_order)
    # print('combinations', combinations)
    #
    # def multiply(a, b):
    #     return [i * j for i, j in zip(a, b)]
    # def multiply(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    #     return


    # for degree in range(2, max_degree+1):
    #     ntriangles = sp.Matrix.hstack(ntriangles, sp.Matrix([i**degree for i in range(1, seq.rows)]), triangle_grid(degree))
    # triangles = sp.Matrix.hstack(*[triangle_grid(degree) for degree in range(1, degree+1)])
    polys = sp.Matrix.hstack(*[comb2act(comb, triangle, multiply_eltw) for comb in combinations])
    # print('polys', polys.shape, polys.__repr__())
    # 1/0

    # middle = triangle['n']
    # print(middle)
    # # print(sp.Matrix.hstack(comb2act(combinations[0], triangle), comb2act(combinations[0], triangle)))
    # for comb in combinations:
    #     print(comb)
    #     # print('n - n', sp.Matrix.hstack(triangle['n'], triangle['a(n-1)']))
    #     print('vals', comb2act(comb, triangle, multiply_eltw) )
    #     middle = sp.Matrix.hstack(middle, comb2act(comb, triangle))
    #     print('middle', middle)
    #     print('polys wanna be', middle[:, 1:])
    #     # print(sp.Matrix.hstack(*[comb2act(comb, triangle)]))
    #
    # print('n_cols', n_cols.shape, n_cols,)
    # print('triangles', triangles.shape, triangles)

    # col_tri = n_cols if library == 'n' else sp.Matrix.hstack(n_cols, triangles)

    # print('n_cols', n_cols.shape, n_cols)
    # print('triangles', triangles.shape, triangles)
    # print('an', seq.shape, seq)
    data = sp.Matrix.hstack(
        seq[1:,:],
        sp.Matrix([1 for _ in range(seq.rows-1)]),  # constant term, i.e. C_0 in a_n = C_0 + C_1*n + C_2*n^2 + ...
        # n_cols,
        # triangles
        # col_tri
        polys
        )
        # sp.Matrix([i for i in range(1, seq.rows)]),
        # triangles)
    # print('data', data.shape, data)

    sol_ref = solution_reference(library, d_max, max_order)
    # print('sol ref', sol_ref)

    if data.cols-1 != len(solution_reference(library, d_max, max_order)):
        raise IndexError(f"Diofantos: Reference (indexing) of solution {solution_reference(library, d_max, max_order)} is not compatible with data (with {data.cols-1} columns).")
    return data, sol_ref


def dataset(seq: list, d_max: int, max_order: int, library: str) -> tuple[sp.Matrix, sp.Matrix, list[str]]:
    """Instant list -> (b, A) for equation discovery / LA system."""

    if max_order < 0:
        raise ValueError("max_order must be > 0. Otherwise needs to be implemented properly.")

    # avoid trivial (empty) system of equations:
    if len(seq)-1 < max_order:
        return sp.Matrix(), sp.Matrix(), []

    data, sol_ref = grid_sympy(sp.Matrix(seq), d_max, max_order, library=library)
    # print('order', max_order)
    # print('data', data)
    # print('data.shape', data.shape)
    # print(str(data), data.__repr__())
    # 1/0

    # if linear or library in ('lin', 'quad', 'cub'):
    #     data = data[:, sp.Matrix([0] + list(i for i in range(2, data.shape[1])))]
    m_limit = 3003
    # print('data', data)
    # b = data[max_order:(max_order + m_limit), 0]
    # A = data[max_order:(max_order + m_limit), 1:]
    if data.shape[0] <= max_order-1:
        raise ValueError(f"max_order ({max_order}) must be < data.nrows ({data.shape[0]}). Otherwise trivial matrix produces trivial solution.")
    b = data[max(0, max_order-1):(max_order + m_limit), 0]
    A = data[max(0, max_order-1):(max_order + m_limit), 1:]
    return b, A, sol_ref


# Test diophantine solver:

# from diophantine import solve
# # # diofant check:
# A = sp.Matrix(
#  [[3, 0 ],
#  [0, 3],
#  [1, 0]])
# # b = sp.Matrix([1.5, 1, 0.5])
# # # solution x=[0.5, 1/3], but no integer one

# # # x = diophantine_solve(A, b)
# # # print('x', x)
# # # # 1/0

# # A = sp.Matrix(
# #     [[3, 0 ],
# #     [0, 3],
# #     [1, 0]])
# # b = sp.Matrix([6, 9, 2])

# # # x = diophantine_solve(A, b)
# # # print('x', x)
# Ainfty = np.array(
#    [[3, 0],
#     [0, 0],
#     [1, 0]])
# # A = Ainfty
# b = np.array([6, 9, 2])
# infty = np.array([6, 0, 2])
# b = infty
# # x = solve(A,b)
# # x = diophantine_solve(A,b)

# # 2 solutions!:
# Ainfty = np.array(
#    [[3, 0, 0],
#     [0, 1, 1],
#     [1, 0, 0]])
# A = Ainfty
# infty = np.array([9, 3, 3])
# b = infty
# x = solve(A,b)
# # x = diophantine_solve(A,b)
# print(x)
# 1/0

# # from diophantine import solve
# # x = diophantine_solve(Ainfty, infty)
# # print('x', x)
# # print(solve(Ainfty, infty))


VERBOSITY = 2  # dev scena
# VERBOSITY = 1  # run scenario


def truth2coeffs(truth: str) -> sp.Matrix:
    """Convert truth from first row of csv into list of coefficients"""
    # print('truth', truth)
    replaced = truth.replace('{', '').replace('}', '')
    peeled = replaced[1:-2] if replaced[-2] == ',' else replaced[1:-1]
    coeffs = peeled.split(',')
    check_coeffs = sp.Matrix(list(int(i) for i in coeffs))
    return check_coeffs


def unnan(seq: list) -> sp.Matrix:
    """Remove nan from sequence."""

    # print(seq)
    seq = sp.Matrix(seq)
    # print(seq)
    if seq.has(sp.nan):
        seq = seq[:list(seq).index(sp.nan), :]
    seq = sp.Matrix([int(i) for i in seq])
    return seq


def unpack_seq(seq_id: str, csv: pd.DataFrame) -> tuple[sp.Matrix, sp.Matrix, str]:
    "Unpack ground truth and terms of the sequence from given csv."

    seq = unnan(csv[seq_id][1:])
    truth = csv[seq_id][0]
    coeffs = truth2coeffs(truth)
    truth = ['a(n) = '] + [f'{str(coeff)}*a(n - {n+1}) + ' for n, coeff in enumerate(coeffs)]
    init_vals = [f'a({n}) = {seq[n]}, ' for n, _ in enumerate(coeffs[:len(seq)])]
    truth = ''.join(truth)[:-3] + ',  \n' + ''.join(init_vals)[:-2]
    return seq, coeffs, truth


def solution_vs_truth(x: sp.Matrix, truth_coeffs: sp.Matrix) -> bool:
    """is_oeis/is_reconst, i.e. return True if solution is identical to the ground truth
    Since Diofantos is in this case executed only with 'non' library, the solution is always
    of the form [c0, c1, ...] corresponding to variables [1, a(n-1), a(n-2), ...].
    """

    nonzero_indices = [i for i in range(len(x)) if (x[i] != 0)]
    if nonzero_indices == []:
        ed_coeffs = []
    elif x[0] != 0:
        ed_coeffs = "equation containing a constant term => non_id (return false)"
    else:
        order = nonzero_indices[-1]
        ed_coeffs = x[1:1 + order, :]
    return ed_coeffs == truth_coeffs


def lib2degrees(library: str) -> int:
    # n_present = 1 if library in ('n', 'nlin', 'nquad', 'ncub') else 0
    degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('n', 'cub', 'ncub') else 'Unknown Library!!'
    n_degree = 0 if library in ('lin', 'quad', 'cub') else degree
    degree = 0 if library == 'n' else degree
    return n_degree, degree


def xlib2orders(x: sp.Matrix, library: str) -> tuple[int, int, int]:
    """Calculate n_present, degree and order out of library string."""

    # n_present = 1 if library in ('n', 'nlin', 'nquad', 'ncub') else 0
    # degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('n', 'cub', 'ncub') else 'Unknown Library!!'
    # n_present = lib2degree(library)[0]
    n_degree = lib2degrees(library)[0]
    degree = lib2degrees(library)[1]
    # print('x', x)
    # print('n_degree', n_degree)
    # print('degree', degree)
    # print('lib', library)
    # # order = (len(x) - degree*n_present)//degree
    order = 0 if degree == 0 else (len(x) - n_degree - 1)//degree
    # if len(x) != degree*n_present + degree*order:
    # print('order', order)
    if len(x) != 1 + n_degree + degree * order:
        raise IndexError('Diofantos: library is not compatible with coefs\' length, i.e. len(x) != degree*n_present + degree*order')

    return  n_degree, degree, order

def lib2verbose(library: str, order: int) -> list[str]:
    """Convert library string into verbose equation."""

    n_degree, degree = lib2degrees(library)

    verbose_eq = (['a(n)'] + ['n']*(n_degree>0) + [f'n^{deg}' for deg in range(2, n_degree+1)] + [f"a(n-{i})" for i in range(1, order + 1)]
        + sum([[f"a(n-{i})^{degree}" for i in range(1, order+1)] for degree in range(2, degree + 1)], []))
    return verbose_eq

def lib2stvars(library: str, order: int) -> list[str]:
    """Convert library string into sympy variables."""

    # n_degree, _ = lib2degrees(library)
    # print('n_degree', n_degree)
    # basis = ['n'] * (n_degree > 0) + [f'a(n-{i})' for i in range(1, order + 1)]
    basis = ['n'] * (library=='n') + [f'a(n-{i})' for i in range(1, order + 1)]
    return basis

def solution2str(x: sp.Matrix, solution_ref: list[str], library: str = None) -> str:
    """Convert solution to string."""

    if x==[]:
        eq = "a(n) = NOT RECONSTRUCTED :-("
    else:
        if solution_ref is None:
            n_degree, degree, order = xlib2orders(x, library)
            # print(n_degree, degree, order)
            # n_present = 1 if library in ('n', 'nlin', 'nquad', 'ncub') else 0
            # degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('n', 'cub', 'ncub') else 'Unknown Library!!'
            # order = (len(x) - degree*n_present)//degree
            # order = 0 if library == 'n' else order
            # if len(x) != degree*n_present + degree*order:
            #     raise IndexError('Diofantos: library is not compatible with coefs\' length, i.e. len(x) != degree*n_present + degree*order')

            # print(degree, order, len(x))
            # print([(i, i%order, i//order) for i in range(1+order+1, 1+order*degree+1)])

            verbose_eq = lib2verbose(library, order)
            verbose_eq = solution_reference(library, order)
        elif library is not None:
            raise ValueError('Diofantos: only one of library or solution_ref can be None, currently both are!')
        else:
            verbose_eq = solution_ref
        # verbose_eq = (['a(n)'] + ['n']*(n_degree>0) + [f'n^{deg}' for deg in range(2, n_degree+1)] + [f"a(n-{i})" for i in range(1, order + 1)]
        #     + sum([[f"a(n-{i})^{degree}" for i in range(1, order+1)] for degree in range(2, degree + 1)], []))
        # print(verbose_eq)
        # print('x', x)
        # 1/0

        verbose_eq = sp.Matrix(verbose_eq)
        # print('verbose_eq', verbose_eq_new.shape, verbose_eq_new)

        # verbose_eq_new = sp.Matrix(['a(n)'] + verbose_eq)
        # verbose_eq = verbose_eq_new
        # print('verbose_eq', verbose_eq_new.shape, verbose_eq_new)
        # print('verbose_eq', verbose_eq.shape, verbose_eq)
        # expr = verbose_eq[:, 1:] * x
        # expr = verbose_eq.dot(x)
        expr = verbose_eq.transpose() * x
        # eq = f"{verbose_eq[0]} = {expr[0]}"
        eq = f"{'a(n)'} = {expr[0]}"
        # print('eq:', eq)
    return eq


def instant_solution_vs_truth(x: sp.Matrix, seq_id: str, csv: pd.DataFrame) -> bool:
    "Instant version of solution_vs_truth to avoid manual extraction of truth from csv."

    _, coeffs, _ = unpack_seq(seq_id, csv)
    return solution_vs_truth(x, coeffs)


def exact_ed(seq_id: str, csv: pd.DataFrame, verbosity: int = VERBOSITY,
             d_max: int = None, max_order: int = None, ground_truth: bool = True, n_of_terms=10**16,
             library: str ='lin') -> tuple[sp.Matrix, str, str, str]:
    # max_order = 25
    # max_order = None
    header = 1 if ground_truth else 0

    # if library == 'lin':
    #     linear = True
    # print('in exact_ed')
    # POTENTIAL ERROR!!!!: WHEN NOT CONVERTING 3.0 INTO 3 FOR SOLVING DIOFANTINE
    if ground_truth:
        seq, coeffs, truth = unpack_seq(seq_id, csv)
        # Handle nans:  # prolly not needed any more:
        if seq.has(sp.nan):
            seq = seq[:list(seq).index(sp.nan), :]
    else:
        seq = unnan(list(csv[seq_id][header:(header + n_of_terms)]))


    # print(seq)
    max_order = sp.floor(seq.rows/2)-1 if max_order is None else max_order

    b, A, sol_ref = dataset(list(seq), d_max, max_order, library=library)
    if sol_ref == []:
        x = []
    else:

        # # print('order', max_order)
        # print(A.shape)
        # print(b.shape)
        # print(b.__repr__())
        # print(A.__repr__())
        # # print(b)
        # # print(A)
        # # 1/0
        # # print('after dataset')

        # m_limit = 3003
        # b = data[max_order:(max_order + m_limit), 0]
        # A = data[max_order:(max_order + m_limit), 1:]
        # b = max_order + m_limit
        # A = sp.Matrix(
        #     [[3, 0],
        #      [0, 3],
        #      [1, 0]])
        # b = sp.Matrix([6, 9, 2])

        if verbosity >= 3:
            print('A, b', A.__repr__(), b.__repr__())
            # print('A[:4][:4] :', A[:6, :6].__repr__(), '\n', A[:, -2].__repr__())
            print('A, b  shapes', A.shape, b.shape)

        x = diophantine_solve(A, b)
    # print(x)
    # if x != []:
    #     print(A*x[0])
    # print('inside exact_ed')
    # print(A)
    # print(x)
    # print(b)
    # print('after sanity check')
    if verbosity >= 3:
        print('x', x)

    if len(x) > 0:
        x = x[0]
        # if linear:
        #     x = sp.Matrix.vstack(sp.Matrix([0]), x)
    # print(x, library)
    eq = solution2str(x, solution_ref=sol_ref, library=None)
    # print(eq)

    # print('x', x)
    if ground_truth:
        return x, sol_ref, eq, coeffs, truth
    else:
        return x, sol_ref, eq, "", ""


def increasing_eed(exact_ed, seq_id: str, csv: pd.DataFrame, verbosity: int = VERBOSITY,
                   d_max: int = None, max_order: int = None, ground_truth: bool = True, n_of_terms=10 ** 16, library: 'str' = None,
                   start_order: int = 1, init: list = None, sindy_hidden: list = None) -> tuple[sp.Matrix, str, str, str, str]:
    """Perform exact_ed with increasing the *max_order* untill the equation that holds (with minimum order) is found."""

    # verbosity = 2
    def eed_step(ed_output, deg_order):
        # print('deg_order', deg_order)
        d_max = deg_order[0]
        order = deg_order[1]
        # print('deg_order', deg_order, len(deg_order), deg_order[2:])
        threshold, ensemble = (None, None) if len(deg_order) == 2 else deg_order[2:]
        # print('threshold, ensemble', threshold, ensemble)

        # print('summary', ed_output)
        # print('eed_step', f'd_max, order: {d_max, order}', seq_id, 'calculating ...')
        if verbosity >= 2:
            print('eed_step', deg_order, seq_id, 'calculating ...')

        # output = ed_output if ed_output[0] != [] else exact_ed(seq_id, csv, verbosity, order, linear, n_of_terms) + (False,)
        # # output = ed_output if ed_output[-1] else exact_ed(seq_id, csv, verbosity, order, linear, n_of_terms) + (False,)
        if not ed_output[-1]:
            if exact_ed.__name__ == 'one_results':
                # print('inc before call', lib, order, ed_output)
                seq = sindy_hidden[d_max-1]

                x, sol_ref, is_reconst, eq, _ = exact_ed(seq, seq_id, csv, None, d_max, order, ground_truth=ground_truth,
                                                         library=library, threshold=threshold, ensemble=ensemble) + (False,)


                # def one_results(seq, seq_id, csv, coeffs, d_max: int, max_order: int,
                #                 threshold: float, library: str, ensemble: bool, library_ensemble: bool = None):

                coefs, truth = '', ''
            else:
                x, sol_ref, eq, coefs, truth, _ = exact_ed(seq_id, csv, verbosity, d_max, order, ground_truth,
                                                           n_of_terms, library=library) + (False,)

            # print('after exact')

            output = x, (sol_ref, d_max, order), eq, coefs, truth, _
            # print(output)
            # print(output[1:])
            if x != []:
                # print('nonempty x')
                if len(x) > 0 and x[-1] == 0 and order >= 2:
                    # print('Unlucky me!' + seq_id + ' The order is lower than maximum although it\'s increasing eed!'
                    #     'this indicates that the equation probably the equation found a loophole in'
                    #     'construction of dataset since it ignores first terms of the sequence.'
                    #     'There may be a way to fix this - not sure but I\'m too lazy to do it now. '
                    #     'I suspect that the equation is not correct for all terms (wrong for the first few).')
                    pass

                # print('inc eed: x vs sol_ref ', len(x), len(sol_ref), x, sol_ref)

                is_check = True
                if sol_order(x, sol_ref)[0] < order:
                    is_check = False
                #     is_check = check_eq_man(x, seq_id, csv, header=ground_truth, n_of_terms=10**5, solution_ref=sol_ref,
                #                             library=None)[0]
                # print('after check x')

                if not is_check:
                    # output = [], "", "", "", False
                    output = ed_output[:-1] + (False,)
                    # print('diofantos failed: non_manual!')
                else:
                    output = output[:-1] + (True,)
                    if verbosity >= 2:
                        print('\nis_check', is_check, 'i.e. found equation!\n')
                    # output = ed_output[:2] + output[2:-1] + (True,)
            else:
                # output = ([],) + (output[1:])
                # output = ed_output[:2] + output[2:]
                pass
        else:
            output = ed_output

        # # x_before = output[0]
        # # nonzero_indices = [i for i in range(len(x)) if (x[i] != 0)]
        # # x = x[:max(nonzero_indices) + 1] if len(nonzero_indices) > 0 else []
        # # # if len(x_before) ,
        # # print('x', x)
        #
        # # if

        # print('output', output)
        # print('after one step of order', order, output)
        return output

    # start = ([], "a(n) = NOT RECONSTRUCTED :-(", "", "", False)
    # start = ([], 'n', solution2str([], library=library[0]), "", "", False)
    start = ([], (None, None, None) , 'a(n) = ?', "", "", False) if init is None else init
    # print(len(start), start)
    d_maxs, orders = range(1, d_max+1), range(start_order, max_order+1)
    # d_maxs, orders = range(3, d_max+1), range(9, max_order+1)
    deg_orders = list(product(d_maxs, orders))
    # deg_orders = product(d_maxs, orders)
    if exact_ed.__name__ == 'one_results':
        deg_orders = product(d_maxs, orders, [i*0.1 for i in range(11)], [False, True])

    # a = list(deg_orders)
    # print('deg_orders', list(deg_orders))

    eed = reduce(eed_step, deg_orders, start)[:-1]
    # print('after reduce')
    # print(eed)

    # for i in range(1, max_order):
    #     if x != []:
    #         x = exact_ed()
    #     else:
    #         return x, eg, ...
    return eed


def eed(x):
    return x>=6


def exp_search(max_order: int = 20, max_found=True, **eed) -> tuple[tuple]:
    """Finds interval (a, b) as a warm start-up for bisection.

    With naive intuitive assumption that large max_order requires tremendous
    computation we try to avoid this by exponentially increasing order and
    starting at 0 order.

    max_found ... True if we already found equation of order _max_order_

    Output: ((a, b), eq)
        - (a, b) for bisection
        - eq ... equation that holds for b
    """

    a = 0
    b = max_order
    # bool = False
    last_order = [max_order] if not max_found else []

    # eed_output = None
    for i in [j ** 2 for j in range(int(math.sqrt(max_order)))] + last_order:
        print(i)
        # if i >= 3:
        #     bool = True


        # eed['max_order'] = i
        eed_output = exact_ed(max_order=i, **eed)
        # eed_output = exact_ed(seq_id, csv, verbosity=VERBOSITY, max_order=i, linear=True, n_of_terms=10 ** 16)

        # if x==[] or
        #     return x, coeffs, eq, truth
        bool = not (eed_output[0] == [] or not check_eq_man(eed_output[0], **eed))
        # bool = not (eed_output[0] == [] or not check_eq_man(eed_output[0], seq_id, csv, n_of_terms=10 ** 8))

        if bool:
            b = i
            print('breaking', i)
            break
        a = i

    return a, b, eed_output


# a, b = exp_search(max_order=20, max_found=False, eed_output=None, csv, seq_id) -> tuple[tuple]:
# print(a, b)
# 1/0


# def bisect(a, b, eed_output):
def bisect(a, b, **eed) -> tuple[tuple]:

    if a == b:
        print('Note bene: bisection unnecessary since a == b')
        return a, eed['eed_output']

    n = 1
    while n <= 22:
        if b - a == 1:
            print(f'winner found: {b}')
            return b, eed['eed_output']
        else:
            c = int((a+b)/2)
            print(c)
            eed_output = exact_ed(max_order=c, **eed)
            if eed_output[0]:
                b = c
            else:
                a = c
        print(f'eof {n}')
        n += 1
    print(f'Bisection unsuccessful!!!')
    RuntimeError('My implementation of bisect() unsuccessful!!!')
    return

# print(bisect(a, b, eed_output, csv, seq_id))
# 1/0


# def adaptive_leed(seq_id, csv, verbosity=VERBOSITY, max_order=20, linear=True, n_of_terms=10**16, max_found=True):
#     """Adapted linear exact ED.
#     I.e. try exact_ed for different orders, since we want as simple
#     equations as possible (e.g. smallest order).
#
#      - max_found ... True if we already found equation of order _max_order_
#     """
#
#     b = None
#     if max_found:
#         x, coeffs, eq, truth = exact_ed(seq_id,
#                                         csv,
#                                         verbosity=verbosity,
#                                         max_order=max_order,
#                                         linear=linear,
#                                         n_of_terms=n_of_terms)
#         if x==[]:
#             return x, coeffs, eq, truth
#         elif not check_eq_man(x, seq_id, csv, n_of_terms=10**8):
#             return x, coeffs, eq, truth
#         else:
#             # I think wrong:
#             b = list(x[1:]).index(0) + 1  # b in bisection
#             eed_output = x, coeffs, ...
#
#
#     b = max_order if b is None else b
#
#     eed = {'seq_id': seq_id, 'csv': csv, 'verbosity': VERBOSITY, 'max_order': 20, 'linear': True, 'n_of_terms': 10 ** 16, 'max_found': True}
#     eed.pop('max_order')
#
#     a, b, exp_output = exp_search(max_order=b, **eed)
#     # if exp_output is None and not max_found:  # No equation found.
#     #     return exp_output
#     # elif exp_output is None:
#     #     return
#
#     # eed_output = eed_output if exp_output is None else exp_output
#
#     eed_output = exp_output
#
#     eed_output = bisect(a, b, eed_output=eed_output, **eed)
#
#
#     # for i in [i**2 for i in range(20)]
#
#
#     # seq = seq[:list(seq).index(sp.nan), :]
#
#     x, coeffs, eq, truth = eed_output
#
#     print('x:', x, x[1:])
#     ed_coeffs = [str(c) for c in x[1:] if c!=0]
#     print('ed_coeffs:', ed_coeffs)
#     print('coeffs:', coeffs)
#     print(coeffs == ed_coeffs)
#     print('eq:', eq)
#     print('truth:', truth)
#
#     return


def check_truth(seq_id: str, csv_filename: str, oeis_friendly=False, library: str = 'nquad'):
    'Check OEIS sequence\'s  website\'s supposed equation against the sequence terms.'

    csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])
    truth = csv[seq_id][0]

    # replaced = truth.replace('{', '').replace('}', '')
    # peeled = replaced[1:-2] if replaced[-2] == ',' else replaced[1:-1]
    # coeffs = peeled.split(',')
    # coeffs = [0] + truth2coeffs(truth)
    # coeffs = [0] + coeffs[:len(csv[seq_id])-2]
    # print(coeffs)
    x = truth2coeffs(truth).row_insert(0, sp.Matrix([0]))
    # x = sp.Matrix(list(int(i) for i in coeffs))
    # print(x)
    sol_ref = solution_reference(library='non', d_max=1, order=x.rows-1)
    is_check = check_eq_man(x, seq_id, csv, n_of_terms=10**5, oeis_friendly=oeis_friendly,
                            solution_ref=sol_ref)
    # print(is_check)
    # is_check = is_check[0]
    # check[]
    return is_check, truth


def sol_order(x: sp.Matrix, solution_ref: list[str]) -> (int, dict):
    # print(order, x)
    if len(x) != len(solution_ref):
        # print(x, solution_ref)
        raise ValueError('Diofantos\' debug: len(x) != len(solution_ref)')
    orders = [i for i in range(1, len(x)+1)]
    # nonzero solution dict:
    x_dict = {var: x[i] for i, var in enumerate(solution_ref) if int(x[i]) != 0}

    # print(x_dict)
    # print([(x_i, var) for x_i, var in x_dict.items()])
    # print('sec22')
    # print([[o for o in orders if str(o) in var] for var, x_i in x_dict.items()])
    # print([max([0] + [o for o in orders if str(o) in var]) for var, x_i in x_dict.items()])
    # print(solution_ref)
    # oo = [0] + [([0] + [o for o in orders], '2' in 'a(n-2)', var, _) for var, _ in x_dict.items()]
    # print(oo)

    return max([0] + [max([0] + [o for o in orders if str(o) in var]) for var, _ in x_dict.items()]), x_dict


def check_eq_man(x: sp.Matrix, seq_id: str, csv: pd.DataFrame,
                 n_of_terms: int = 500, header: bool = True,
                 oeis_friendly=0, solution_ref: list[str] = None, library: str = None) -> (bool, int):
    """Manually check if exact ED returns correct solution, i.e. recursive equation."""
    if not x:
    # if x==[]:
        return False, "no reconst", "no reconst"
    n_of_terms = max(n_of_terms, len(x))
    # print('header', header)
    header = 1 if header else 0
    # print('header', header)
    seq = unnan(csv[seq_id][header:n_of_terms])
    # print('seq', seq)
    seq = sp.Matrix(list(reversed(seq[:])))
    # print(seq[-10:])
    # 1/0

    # seq = sp.Matrix(csv[seq_id][header:])[:n_of_terms, :]
    # # Handle nans:
    # if seq.has(sp.nan):
    #     seq = seq[:list(seq).index(sp.nan), :]

    # # Why? In order to set the order of the recursion to the minimal and thus take as few first terms as possible:
    # nonzero_indices = [i for i in range(len(x)) if (x[i] != 0)]
    # if nonzero_indices == []:
    #     x = [0, 0]
    #     # print(type(seq), type(seq[0, :]))
    #     fake_reconst = seq[0, :] + sp.Matrix([1])
    # elif nonzero_indices == [0]:
    #     x = [x[0]] + [0]
    #     fake_reconst = seq[0, :] + sp.Matrix([1])
    # else:
    #     x = x[:max(nonzero_indices) + 1]
    #     fake_reconst = []
    # x = sp.Matrix(x)

    # print('sol_ref out', solution_ref)

    if solution_ref is None:
        n_degree, degree, order = xlib2orders(x, library)
        raise(NotImplementedError('solution_ref is None when checking out the solution!!!'))
        # print('n_degree, degree, order:', n_degree, degree, order)
        solution_ref = solution_reference(library, order)

    # solution_ref = solution_ref[1:]  # len(x) = len(sol_ref)

    order_, x_dict = sol_order(x, solution_ref)
    # print('order, x_dict', order_, x_dict)
    # 1/0
    # print('x', x)
    # print('sol_ref', solution_ref)
    # if order == 0:
    #     1/0


    def stvar2term(stvar: str, till_now: sp.Matrix) -> int:
        # print('in stvar2term', stvar)
        if stvar == '1': return 1
        elif stvar == 'n': return till_now.rows
        else:
            # print('\n\nstvar!!!: ', stvar, order, x_dict, '\n\n')
            order2 = ([i for i in range(order_+1) if stvar == f'a(n-{i})'] + [0])[0]
            # print('in stvar2term', order_, order2, stvar, till_now, len(till_now))
            ret = till_now[order2-1]
            # print(order2 - 1)
            # print('endof stvar2term', stvar, 'result', ret, 'defy', till_now[1])
            return ret  # (order - 1) + 1 ... i.e. first 1 from list indexing, second 1 from constant term


    def var2term(var: str, till_now: sp.Matrix) -> int:
        comb = var.split('*')
        # comb2act(comb, x_dict, lambda x,y: x*y)
        def updateit(current, elt):
            # print('in updateit', current, elt, var, till_now, solution_ref)
            # print('in updateit in', till_now.shape[0])
            # print('\ncurent magnitude', len(str(current)), '\n')
            ret = current*stvar2term(elt, till_now)
            # print(ret)
            return ret

        ret = reduce(updateit, comb, 1)
        # print('returned var2term')
        return ret


        # remnants = {stvar: stvar for i in range(len(x)) if x[i] != 0}
        # {str(var.split('*')): i for i, var in enumerate(x): }

    # n_present = 1 if library in ('n', 'nlin', 'nquad', 'ncub') else 0
    # degree = 0 if library == 'n' else 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('cub', 'ncub') else 'Unknown Library!!'
    # order = (len(x) - degree*n_present)//degree
    # order = 0 if library == 'n' else order
    # if len(x) != degree*n_present + degree*order:
    #     raise IndexError('Diofantos: library is not compatible with coefs\' length, i.e. len(x) != degree*n_present + degree*order')
    # # print(degree, order, len(x))

    # print(n_degree, degree, order, x, seq_id, library)

    def an(till_now: sp.Matrix) -> sp.Matrix:

        # print('till_now, x', till_now, x)
        # print('till_now, x', till_now.shape[0], x)
        if till_now.shape[0] > 300:
            raise RecursionError('Diofantos: Recursion limit reached. Try to increase it.')
        # print(type(x), x[0])
        # print('inside')

        # print('before anext')
        # anext = sp.Matrix([sum([x_i*var2term(var, till_now) for var, x_i in x_dict.items()])])
        ane = 0
        for var, x_i in x_dict.items():
            # if len(till_now) > 16:
                # print('len(till_now)', len(till_now))
                # print('var, x_i:', var, x_i, 'var2term(var, till_now)', var2term(var, till_now))
            ane += x_i*var2term(var, till_now)

        anext = sp.Matrix([ane])
        # anext = sp.Matrix([sum(ane)])
        # anext = sp.Matrix([sum([x_i * var2term(var, till_now)

        # if len(till_now) > 10:
            # print('after anext')
            # print(anext)
        # 1/0
        return anext

    # if fake_reconst != []:
    #     reconst = an(fake_reconst, x)
    # else:
    #     reconst = seq[:max(order, min(oeis_friendly, len(seq))), :]  # init reconst

    # reconst = sp.Matrix(reversed(seq[:max(order, min(oeis_friendly, len(seq)))]))  # init reconst
    # reconst = sp.Matrix(list(reversed(seq[:max(order, min(oeis_friendly, len(seq)))])))
    # print(order)

    # print(oeis_friendly, len(seq), order_, list(reversed(list(seq)))[:5])

    # print(order_, oeis_friendly, len(seq))
    # print(max(order_, min(oeis_friendly, len(seq))))
    # print(seq[-2:, :])
    # 1/0

    reconst = seq[-max(order_, min(oeis_friendly, len(seq))):, :] if order_ != 0 else sp.Matrix([an(sp.Matrix([]))])


    # print('reconst', reconst)
    # 1/0
    # if [i for i in x[n_degree:] if i != 0] == []:
    #     reconst = an(sp.Matrix([0 for i in reconst[:]]), x)
        # reconst = sp.Matrix([])

    # # reconst = list(seq[:max(order, min(oeis_friendly, len(seq)))])
    # # print(type(reconst))
    # print(reconst)
    # print(len(reconst))
    #
    # # 1/0
    # # else:
    # #     reconst = seq[:len(x)-1, :]  # init reconst
    # # print('reconst:', reconst)
    # print('reconst', reconst)

    # print(-1, list(reversed(reconst))[:20])
    for i in range(len(seq) - len(reconst)):
        # reconst = reconst.col_join(sp.Matrix([an(reconst, x)]))
        # reconst = reconst.col_join(an(reconst, x))
        reconst = an(reconst).col_join(reconst)
        # if i > 10:
        #     print(i, list(reversed(reconst))[:20])
        #     # 1/0

    # 1/0
    # print(reconst)
    # print(seq)
    # print(type(reversed(reconst[:])))
    # print(type(reversed(seq[:])))
    out = reconst == seq, list(reversed(reconst[:])), list(reversed(seq[:]))
    # print(len(reconst), len(seq))
    # print([(n, i) for n, i in enumerate(seq) if i != reconst[n]])
    # print(out[0])
    # print(out[1][:20])
    # print(out[2][:20])

    # if order == 0:
    #     print('\n'*8, 'tle')
    # if order == 0:
    #     print('\n\norder 0!, ans:', out[0], out[1])
    #     print('order 0!, ans:', out[2], '\n\n')
    #     # 1/0
    return out

if __name__ == '__main__':
    # from proged times:
    # has_titles = 1
    # csv = pd.read_csv('oeis_selection.csv')[has_titles:]

    # csv = pd.read_csv('linear_database.csv', low_memory=False)
    csvfilename = 'linear_database_full.csv'
    if os.getcwd()[-11:] == 'ProGED_oeis':
        csvfilename = 'ProGED_oeis/examples/oeis/linear_database_full.csv'

    # csv = pd.read_csv('linear_database_full.csv', low_memory=False)
    csv = pd.read_csv(csvfilename, low_memory=False)
    # print(csv.columns)

    # # eq = exact_ed("A000045", csv)
    # # x, eq, truth = exact_ed("A000004", csv, n_of_terms=30)
    # adaptive_leed("A152185", csv, max_order=20)

    x = sp.Matrix([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1])
    tuple_ = (6, -6, -19, 24, 24, -19, -6, 6, -1)
    tuple_ = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    tuple_ = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1)
    tuple_ = (0, 0, 1, 1, 0, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 0, 1, 1, 0, 0, -1)
    tuple_ = (3, -2, 0, 0, 0, -1, 1)  #  'A356621'
    tuple_ = (1, 0, 0, 1, -1)  #  'A356621'
    tuple_ = (3, -2, -1, 0, 1, 2, -3, 1)
    tuple_ = (3, -1, -4, 4, -4, 5, 1, -5, 6, -10, 8, -8, 10, -6, 5, -1, -5, 4, -4, 4, 1, -3, 1)
    tuple_ = (17, -114, 348, -228, -1524, 3888, -216, -11046, 11382, 12012, -26544, 84, 28812, -13152, -15816, 13407, 3201, -5834, 628, 984, -288)

    x = sp.Matrix([0] + list(tuple_))
    id_ = 'A025858'
    id_ = 'A246175'
    id_ = 'A025924'  # george fisher 24 september 2022
    id_ = 'A029252'
    # ({0, 0, 1, 1, 0, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 0, 1, 1, 0, 0, -1): A029252
    id_ = 'A356621'
    id_ = 'A026471'
    id_ = 'A057524'
    id_ = 'A296999'
    id_ = 'A065025'

    # ['A296999', 'A235933', 'A166986', 'A293981', 'A335247', 'A003733', 'A212578', 'A092634', 'A135619', 'A133058',
    #  'A242350', 'A105067', 'A182141', 'A117154', 'A108792', 'A293979', 'A170729', 'A100774', 'A075412', 'A087099',
    #  'A324472', 'A337241', 'A118536', 'A341893', 'A224808', 'A003997', 'A341895', 'A044941', 'A003999', 'A214394',
    #  'A176646', 'A293978', 'A030132', 'A194768', 'A194769', 'A098616', 'A249668', 'A160769', 'A173908', 'A339852',
    #  'A133679', 'A109303', 'A007752', 'A170775', 'A346054', 'A247486', 'A003330', 'A008820', 'A306245', 'A071435',
    #  'A157069']
    # ['A057524', 'A164009']

    id_ = 'A000045'
    x = sp.Matrix([0, 1, 1, 0, 0, 0, 0])
    # x = sp.Matrix([1, 1])
    x = sp.Matrix([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    # x = sp.Matrix([0, 0, 0, 1, 1, 0, 0, 0, 0])
    # x = sp.Matrix([1, 1, 0, 0, 0, 0])
    # is_check = check_eq_man(x, id_, csv, library='lin')
    # is_check = check_eq_man(x, id_, csv, library='nlin')
    is_check = check_eq_man(x, id_, csv, library='nquad')
    # is_check = check_eq_man(x, id_, csv, library='ncub')
    # is_check = check_eq_man(x, id_, csv, library='cub')
    print(is_check[0])
    print(is_check[1][:20])
    print(is_check[2][:20])
    # print([i[:20] for i in is_check[2]])
    print('ehere')
    1/0
    # is_check = check_eq_man(x, id_, csv, oeis_friendly=34)
    print(is_check[0])
    print(list(is_check[1]))
    print(list(is_check[2]))
    print(list(is_check[1])[20:120])
    print(list(is_check[2])[20:120])
    print('solution2str', solution2str(sp.Matrix([2, 3,1,1,0,5]), 2))
    print('solution2str', solution2str(sp.Matrix([2,1,1,5,5,623, 57,6, 76,6, 6, 76, ]), 3))


