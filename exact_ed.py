"""Discover equation(s) on OEIS sequence to discover direct, recursive or even direct-recursive equation.
"""

import os
import sympy as sp
import pandas as pd
import time
import math
# from typing import Union
from functools import reduce


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
def grid_sympy(seq: sp.MutableDenseMatrix, max_order: int, library: str):  # seq.shape=(N, 1)
    # seq = seq if nof_eqs is None else seq[:nof_eqs]
    # seq = seq[:nof_eqs, :]
    # seq = seq[:shape[0]-1, :]
    # n = len(seq)

    max_degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('cub', 'ncub') else 'Unknown Library!!'
    n_cols = sp.Matrix.hstack(*[sp.Matrix([n**degree for n in range(1, seq.rows)]) for degree in range(1, max_degree+1)]) if library in ('nlin', 'nquad', 'ncub') else sp.Matrix()
    # ntriangles = triangle_grid(1)

    def triangle_grid(max_degree):
        return sp.Matrix(seq.rows-1, max_order,
                        (lambda i,j: (seq[max(i-j,0)]**max_degree)*(1 if i>=j else 0))
                        )
    # ntriangles = sp.Matrix.hstack(sp.Matrix([i for i in range(1, seq.rows)]), triangle_grid(1))

    # for degree in range(2, max_degree+1):
    #     ntriangles = sp.Matrix.hstack(ntriangles, sp.Matrix([i**degree for i in range(1, seq.rows)]), triangle_grid(degree))
    triangles = sp.Matrix.hstack(*[triangle_grid(degree) for degree in range(1, max_degree+1)])

    col_tri = sp.Matrix.hstack(n_cols, triangles)

    # print('n_cols', n_cols.shape, n_cols)
    # print('triangles', triangles.shape, triangles)
    # print('an', seq.shape, seq)
    data = sp.Matrix.hstack(
        seq[1:,:],
        # n_cols,
        # triangles
        col_tri
        )
        # sp.Matrix([i for i in range(1, seq.rows)]),
        # triangles)
    return data

def dataset(seq: list, max_order: int, linear: bool, library: str):
    "Instant list -> (b, A) for equation discovery / LA system."

    if max_order <= 0:
        raise ValueError("max_order must be > 0. Otherwise needs to be implemented properly.")

    data = grid_sympy(sp.Matrix(seq), max_order, library=library)
    # print('order', max_order)
    # print('data', data)

    # if linear or library in ('lin', 'quad', 'cub'):
    #     data = data[:, sp.Matrix([0] + list(i for i in range(2, data.shape[1])))]
    m_limit = 3003
    # print('data', data)
    # b = data[max_order:(max_order + m_limit), 0]
    # A = data[max_order:(max_order + m_limit), 1:]
    b = data[max_order-1:(max_order + m_limit), 0]
    A = data[max_order-1:(max_order + m_limit), 1:]
    return b, A


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
    "is_oeis/is_reconst, i.e. return True if solution is identical to the ground truth"

    nonzero_indices = [i for i in range(len(x)) if (x[i] != 0)]
    if nonzero_indices == []:
        ed_coeffs = []
    elif x[0] != 0:
        ed_coeffs = "containing non-recursive n-term"
    else:
        order = nonzero_indices[-1]
        ed_coeffs = x[1:1 + order, :]
    return ed_coeffs == truth_coeffs


def solution2str(x: sp.Matrix, library: str) -> str:
    "Convert solution to string."

    n_present = 1 if library in ('nlin', 'nquad', 'ncub') else 0
    degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('cub', 'ncub') else 'Unknown Library!!'
    order = (len(x) - degree*n_present)//degree
    if len(x) != degree*n_present + degree*order:
        raise IndexError('Diofantos: library is not compatible with coefs\' length, i.e. len(x) != degree*n_present + degree*order')

    # print(degree, order, len(x))
    # print([(i, i%order, i//order) for i in range(1+order+1, 1+order*degree+1)])

    verbose_eq = (['a(n)', 'n'] + [f'n^{degree}' for degree in range(2, degree+1)] + [f"a(n-{i})" for i in range(1, order + 1)]
        + sum([[f"a(n-{i})^{degree}" for i in range(1, order+1)] for degree in range(2, degree + 1)], []))
    # print(verbose_eq)
    # 1/0

    verbose_eq = sp.Matrix([verbose_eq])
    if x==[]:
        eq = "a(n) = NOT RECONSTRUCTED :-("
    else:
        expr = verbose_eq[:, 1:] * x
        eq = f"{verbose_eq[0]} = {expr[0]}"
    return eq


def instant_solution_vs_truth(x: sp.Matrix, seq_id: str, csv: pd.DataFrame) -> bool:
    "Instant version of solution_vs_truth to avoid manual extraction of truth from csv."

    _, coeffs, _ = unpack_seq(seq_id, csv)
    return solution_vs_truth(x, coeffs)


def exact_ed(seq_id: str, csv: pd.DataFrame, verbosity: int = VERBOSITY,
             max_order: int = None, linear: bool = True, n_of_terms=10**16, library: str ='lin') -> tuple[sp.Matrix, str, str, str]:
    # max_order = 25
    # max_order = None
    header = 1 if linear else 0

    if library == 'lin':
        linear = True
    # print('in exact_ed')
    # POTENTIAL ERROR!!!!: WHEN NOT CONVERTING 3.0 INTO 3 FOR SOLVING DIOFANTINE
    if linear:
        seq, coeffs, truth = unpack_seq(seq_id, csv)
        # Handle nans:
        if seq.has(sp.nan):
            seq = seq[:list(seq).index(sp.nan), :]
    else:
        seq = unnan(list(csv[seq_id][header:(header + n_of_terms)]))


    # print(seq)
    max_order = sp.floor(seq.rows/2)-1 if max_order is None else max_order
    b, A = dataset(list(seq), max_order, linear=linear, library=library)

    # print('order', max_order)
    # print(A.shape)
    # print(b.shape)
    # print(b)
    # print(A)
    # 1/0
    # print('after dataset')

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
    # print(A*x[0])
    # print(b)
    # print('after sanity check')
    if verbosity >= 3:
        print('x', x)

    if len(x) > 0:
        x = x[0]
        if linear:
            x = sp.Matrix.vstack(sp.Matrix([0]), x)
    eq = solution2str(x, library)

    # print('x', x)
    if linear:
        return x, eq, coeffs, truth
    else:
        return x, eq, "", ""


def increasing_eed(seq_id: str, csv: pd.DataFrame, verbosity: int = VERBOSITY,
                   max_order: int = None, linear: bool = True, n_of_terms=10 ** 16, library: str = 'lin',
                   start_order: int = 1) -> tuple[sp.Matrix, str, str, str]:
    """Perform exact_ed with increasing the *max_order* untill the equation that holds (with minimum order) is found."""

    # verbosity = 2
    def eed_step(ed_output, order):
        # print('summary', ed_output)
        # print('eed_step', order, seq_id, 'calculating ...')
        if verbosity >= 2:
            print('eed_step', order, seq_id, 'calculating ...')

        # output = ed_output if ed_output[0] != [] else exact_ed(seq_id, csv, verbosity, order, linear, n_of_terms) + (False,)
        # # output = ed_output if ed_output[-1] else exact_ed(seq_id, csv, verbosity, order, linear, n_of_terms) + (False,)
        if not ed_output[-1]:
            output = exact_ed(seq_id, csv, verbosity, order, linear, n_of_terms, library=library) + ( False,)
            # print(output)
            # print(output[1:])
            if output[0] != []:
                if len(output[0]) > 0 and output[0][-1] == 0 and order >= 2:
                    # print('Unlucky me!' + seq_id + ' The order is lower than maximum although it\'s increasing eed!'
                    #     'this indicates that the equation probably the equation found a loophole in'
                    #     'construction of dataset since it ignores first terms of the sequence.'
                    #     'There may be a way to fix this - not sure but I\'m too lazy to do it now. '
                    #     'I suspect that the equation is not correct for all terms (wrong for the first few).')
                    pass

                is_check = check_eq_man(output[0], seq_id, csv, n_of_terms=10**5, library=library)[0]
                if not is_check:
                    # output = [], "", "", "", False
                    output = ed_output[:2] + output[2:]
                else:
                    output = output[:-1] + (True,)
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

        # print('after one step of order', order, output)
        return output

    # start = ([], "a(n) = NOT RECONSTRUCTED :-(", "", "", False)
    start = ([], solution2str([], library=library), "", "", False)
    orders = range(start_order, max_order+1)

    eed = reduce(eed_step, orders, start)[:4]
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


def adaptive_leed(seq_id, csv, verbosity=VERBOSITY, max_order=20, linear=True, n_of_terms=10**16, max_found=True):
    """Adapted linear exact ED.
    I.e. try exact_ed for different orders, since we want as simple
    equations as possible (e.g. smallest order).

     - max_found ... True if we already found equation of order _max_order_
    """

    b = None
    if max_found:
        x, coeffs, eq, truth = exact_ed(seq_id,
                                        csv,
                                        verbosity=verbosity,
                                        max_order=max_order,
                                        linear=linear,
                                        n_of_terms=n_of_terms)
        if x==[]:
            return x, coeffs, eq, truth
        elif not check_eq_man(x, seq_id, csv, n_of_terms=10**8):
            return x, coeffs, eq, truth
        else:
            # I think wrong:
            b = list(x[1:]).index(0) + 1  # b in bisection
            eed_output = x, coeffs, ...


    b = max_order if b is None else b

    eed = {'seq_id': seq_id, 'csv': csv, 'verbosity': VERBOSITY, 'max_order': 20, 'linear': True, 'n_of_terms': 10 ** 16, 'max_found': True}
    eed.pop('max_order')

    a, b, exp_output = exp_search(max_order=b, **eed)
    # if exp_output is None and not max_found:  # No equation found.
    #     return exp_output
    # elif exp_output is None:
    #     return

    # eed_output = eed_output if exp_output is None else exp_output

    eed_output = exp_output

    eed_output = bisect(a, b, eed_output=eed_output, **eed)


    # for i in [i**2 for i in range(20)]


    # seq = seq[:list(seq).index(sp.nan), :]

    x, coeffs, eq, truth = eed_output

    print('x:', x, x[1:])
    ed_coeffs = [str(c) for c in x[1:] if c!=0]
    print('ed_coeffs:', ed_coeffs)
    print('coeffs:', coeffs)
    print(coeffs == ed_coeffs)
    print('eq:', eq)
    print('truth:', truth)

    return


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
    is_check = check_eq_man(x, seq_id, csv, n_of_terms=10**5, oeis_friendly=oeis_friendly, library=library)
    # print(is_check)
    # is_check = is_check[0]
    # check[]
    return is_check, truth


def check_eq_man(x: sp.Matrix, seq_id: str, csv: str,
                 n_of_terms: int = 500, header: bool = True,
                 oeis_friendly=0, library: str = 'nlin') -> (bool, int):
    """Manually check if exact ED returns correct solution, i.e. recursive equation."""
    if not x:
    # if x==[]:
        return False, "no reconst", "no reconst"
    n_of_terms = max(n_of_terms, len(x))
    header = 1 if header else 0
    seq = unnan(csv[seq_id][header:n_of_terms])
    seq = sp.Matrix(list(reversed(seq[:])))
    # print(seq[-10:])

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

    n_present = 1 if library in ('nlin', 'nquad', 'ncub') else 0
    degree = 1 if library in ('lin', 'nlin') else 2 if library in ('quad', 'nquad') else 3 if library in ('cub', 'ncub') else 'Unknown Library!!'
    order = (len(x) - degree*n_present)//degree
    if len(x) != degree*n_present + degree*order:
        raise IndexError('Diofantos: library is not compatible with coefs\' length, i.e. len(x) != degree*n_present + degree*order')
    # print(degree, order, len(x))

    def an(till_now: sp.Matrix, x: sp.Matrix) -> sp.Matrix:
        # print('till_now, x', till_now, x)
        # print(type(x), x[0])
        coefs = x[:]
        coefs.reverse()
        # print(type(coefs))
        # out = sp.Matrix([coefs[-1]*till_now.rows]) + till_now[-len(coefs[:-1]):, :].transpose()*sp.Matrix(coefs[:-1])

        # for d in range(1, degree+1):
        #     a += till_now[-d]**d

        # order = (len(coefs)-degree*n_present)//degree
        # print('inside')

        coefs = sp.Matrix(coefs)
        a = 0
        # print('til_now:', till_now)
        # print('coefs:', coefs)
        # a = till_now[0]*coefs[-1]*n_present + till_now[-order:, :].dot(coefs[-(order+degree*n_present):-degree*n_present, :])
        # print('a:', a)

        for d in range(1, degree+1):
            # a.applyfunc(lambda x: x**d).transpose()*coefs[-order*d:-order*(d-1), :]
            # a = (a.multiply_elementwisel(
            # a += (till_now[-d])**d*coefs[-d]*n_present
            a += x[d-1]**d*n_present
            # print('a: pred', a)
            # a += till_now[-(order*d+degree):-(order*(d-1)-degree), :].applyfunc( lambda x: x ** d).dot(coefs[-order * d:-order * (d - 1), :])
            # print(order, d, degree, n_present)
            # print(till_now[(order*(d-1)):(order*d), :], x[(order*(d-1)+degree*n_present):(order*(d)+degree*n_present), :])
            a += till_now[:order, :].applyfunc( lambda x: x ** d).dot(x[(order*(d-1)+degree*n_present):(order*(d)+degree*n_present), :])
            # print(till_now[:order, :])
            # print(till_now[:order, :].applyfunc( lambda x: x ** d))
            # print(x[(order*(d-1)+degree*n_present):(order*(d)+degree*n_present), :])
            # 1/0

            # .applyfunc( lambda x: x ** d).dot(x[(order*(d-1)+degree*n_present):(order*(d)+degree*n_present), :]))
            # print('a:', a)
            # print(till_now[-(order*d):-(order*(d-1)), :])
            # print(till_now[-(order*d):-(order*(d-1)+degree), :].applyfunc( lambda x: x ** d))
            # 1/0
            # .transpose() * sp.Matrix(
                # coefs[:-1]))

            # for i in range(1, order+1):
            #     a += till_now[-i]**i * till_now[-d+i]**(d-i)
            # a += till_now[-d]**d

        # print('a:', a)
        # a = till_now [k]
        # out =  sp.Matrix([coefs[-1]*(till_now.rows-1)]) + till_now[-len(coefs[:-1]):, :].transpose()*sp.Matrix(coefs[:-1])
        # print(a)
        return sp.Matrix([a])
        # return (x[0] * till_now.rows + till_now[-len(x[1:]):, :].transpose() * x[1:, :])[0]

    # if fake_reconst != []:
    #     reconst = an(fake_reconst, x)
    # else:
    #     reconst = seq[:max(order, min(oeis_friendly, len(seq))), :]  # init reconst

    # reconst = sp.Matrix(reversed(seq[:max(order, min(oeis_friendly, len(seq)))]))  # init reconst
    # reconst = sp.Matrix(list(reversed(seq[:max(order, min(oeis_friendly, len(seq)))])))
    # print(order)
    reconst = seq[-max(order, min(oeis_friendly, len(seq))):, :]
    # reconst = list(seq[:max(order, min(oeis_friendly, len(seq)))])
    # print(type(reconst))
    # print(reconst)
    # print(len(reconst))
    #
    # print(order, degree, x, seq_id, library, reconst, seq)
    # 1/0
    # else:
    #     reconst = seq[:len(x)-1, :]  # init reconst

    for i in range(len(seq) - len(reconst) - 0):
        # reconst = reconst.col_join(sp.Matrix([an(reconst, x)]))
        # reconst = reconst.col_join(an(reconst, x))
        reconst = an(reconst, x).col_join(reconst)
        # print(i, list(reversed(reconst))[:20])

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


