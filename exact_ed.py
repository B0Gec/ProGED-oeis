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
def grid_sympy(seq: sp.MutableDenseMatrix, max_order: int):  # seq.shape=(N, 1)
    # seq = seq if nof_eqs is None else seq[:nof_eqs]
    # seq = seq[:nof_eqs, :]
    # seq = seq[:shape[0]-1, :]
    # n = len(seq)
    indexes_sympy_uncut = sp.Matrix(seq.rows-1,
                                    max_order,
                                    (lambda i,j: (seq[max(i-j,0)])*(1 if i>=j else 0))
                                    )
    data = sp.Matrix.hstack(
        seq[1:,:],
        sp.Matrix([i for i in range(1, seq.rows)]),
        indexes_sympy_uncut)
    return data

def dataset(seq: list, max_order: int, linear: bool):
    "Instant list -> (b, A) for equation discovery / LA system."

    data = grid_sympy(sp.Matrix(seq), max_order)
    if linear:
        data = data[:, sp.Matrix([0] + list(i for i in range(2, data.shape[1])))]
    m_limit = 3003
    b = data[max_order:(max_order + m_limit), 0]
    A = data[max_order:(max_order + m_limit), 1:]
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


# VERBOSITY = 2  # dev scena
VERBOSITY = 1  # run scenario


def truth2coeffs(truth: str):
    """Convert truth from first row of csv into list of coefficients"""
    replaced = truth.replace('{', '').replace('}', '')
    peeled = replaced[1:-2] if replaced[-2] == ',' else replaced[1:-1]
    coeffs = peeled.split(',')
    check_coeffs = sp.Matrix(list(int(i) for i in coeffs))
    return check_coeffs


def unpack_seq(seq_id: str, csv: pd.DataFrame):
    "Unpack ground truth and terms of the sequence from given csv."

    seq = sp.Matrix(csv[seq_id][1:])
    if seq.has(sp.nan):
        seq = seq[:list(seq).index(sp.nan), :]
    truth = csv[seq_id][0]
    coeffs = truth2coeffs(truth)
    truth = ['a(n) = '] + [f'{str(coeff)}*a(n - {n+1}) + ' for n, coeff in enumerate(coeffs)]
    init_vals = [f'a({n}) = {seq[n]}, ' for n, _ in enumerate(coeffs[:len(seq)])]
    truth = ''.join(truth)[:-3] + ',  \n' + ''.join(init_vals)[:-2]
    return seq, coeffs, truth


def solution_vs_truth(x: sp.Matrix, truth_coeffs: sp.Matrix):
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


def solution2str(x: sp.Matrix):
    verbose_eq = ['a(n)', 'n'] + [f"a(n-{i+1})" for i in range(len(x)-1)]
    verbose_eq = sp.Matrix([verbose_eq])
    if x==[]:
        eq = "a(n) = NOT RECONSTRUCTED :-("
    else:
        expr = verbose_eq[:, 1:] * x
        eq = f"{verbose_eq[0]} = {expr[0]}"
    return eq


def instant_solution_vs_truth(x: sp.Matrix, seq_id: str, csv: pd.DataFrame):
    "Instant version of solution_vs_truth to avoid manual extraction of truth from csv."

    _, coeffs, _ = unpack_seq(seq_id, csv)
    return solution_vs_truth(x, coeffs)


def exact_ed(seq_id: str, csv: pd.DataFrame, verbosity: int = VERBOSITY,
             max_order: int = None, linear: bool = True, n_of_terms=10**16):
    # max_order = 25
    # max_order = None
    header = 1 if linear else 0

    # POTENTIAL ERROR!!!!: WHEN NOT CONVERTING 3.0 INTO 3 FOR SOLVING DIOFANTINE
    if linear:
        seq, coeffs, truth = unpack_seq(seq_id, csv)
        # Handle nans:
        if seq.has(sp.nan):
            seq = seq[:list(seq).index(sp.nan), :]
    else:
        seq = sp.Matrix(csv[seq_id][header:(header + n_of_terms)])

    # if linear:
    #     # truth = '(-34,45,1, -35, 8)'
    #     truth = csv[seq_id][0]
    #     # print(f'truth:{truth}')
    #     # coeffs = truth[1:-1].split(',')[:min(n_of_terms, len(seq))]
    #     coeffs = truth2coeffs(truth)

    max_order = sp.floor(seq.rows/2)-1 if max_order is None else max_order
    # data = grid_sympy(seq, max_order)
    # if linear:
    #     data = data[:, sp.Matrix([0] + list(i for i in range(2, data.shape[1])))]
    # b, A = dataset(seq, max_order, linear=linear)
    b, A = dataset(list(seq), max_order, linear=linear)
    print(A.shape)

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
        print('A[:4][:4] :', A[:6, :6].__repr__(), '\n', A[:, -2].__repr__())
        print('A, b  shapes', A.shape, b.shape)

    x = diophantine_solve(A, b)
    if verbosity >= 3:
        print('x', x)
    # verbose_eq = ['a(n)', 'n']
    #
    # for i in range(max_order):
    #     verbose_eq += [f"a(n-{i+1})"]
    # verbose_eq = sp.Matrix([verbose_eq])
    # # print('--- csv to linear truth ok:', truth, coeffs)

    # if linear:
    #     truth = ['a(n) = '] + [f'{str(coeff)}*a(n - {n+1}) + ' for n, coeff in enumerate(coeffs)]
    #     init_vals = [f'a({n}) = {seq[n]}, ' for n, _ in enumerate(coeffs[:len(seq)])]
    #     # print(f'truth: {truth}')
    #     truth = ''.join(truth)[:-3] + ',  \n' + ''.join(init_vals)[:-2]
    #     if verbosity >= 2:
    #         print(f'truth: {truth}')
    #     # print(seq[:len(coeffs)])

    if x==[]:
        if verbosity >= 2:
            print('NO EQS FOUND!!!')
        eq = "a(n) = NOT RECONSTRUCTED :-("
        # 1/0
    else:
        if verbosity >= 2:
            print('We found an equation!!!:')
        x = x[0]
        if linear:
            x = sp.Matrix.vstack(sp.Matrix([0]), x)
        eq = solution2str(x)
        # expr = verbose_eq[:, 1:] * x
        # eq = f"{verbose_eq[0]} = {expr[0]}"
        if verbosity >= 2:
            print('eq: ', eq)
        # x = eq

    if linear:
        return x, eq, coeffs, truth
    else:
        return x, eq, "", ""


def increasing_eed(seq_id: str, csv: pd.DataFrame, verbosity: int = VERBOSITY,
                   max_order: int = None, linear: bool = True, n_of_terms=10 ** 16):
    """Perform exact_ed with increasing the *max_order* untill the equation that holds (with minimum order) is found."""

    def eed_step(ed_output, order):
        # print('summary', ed_output, order)
        # x = ed_output[0]
        # if x != []:
        #     return ed_output
        # else:
        #     print('tle meljem')
        #     ed_out  = exact_ed(seq_id, csv, verbosity,
        #                  order, linear, n_of_terms)
        #     return ed_out
        output = ed_output if ed_output[0] != [] else exact_ed(seq_id, csv, verbosity, order, linear, n_of_terms)
        return output

    start = ([], "", "", "")
    orders = range(1, max_order)

    eed = reduce(eed_step, orders, start)
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


def check_truth(seq_id: str, csv_filename: str, oeis_friendly=False):
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
    is_check = check_eq_man(x, seq_id, csv, n_of_terms=10**5, oeis_friendly=oeis_friendly)
    # print(is_check)
    # is_check = is_check[0]
    # check[]
    return is_check, truth


def check_eq_man(x: sp.Matrix, seq_id: str, csv: str,
                 n_of_terms: int = 500, header: bool = True,
                 oeis_friendly=0) -> (bool, int):
    """Manually check if exact ED returns correct solution, i.e. recursive equation."""
    if not x:
    # if x==[]:
        return False, "no reconst", "no reconst"
    n_of_terms = max(n_of_terms, len(x))
    header = 1 if header else 0
    seq = sp.Matrix(csv[seq_id][header:])[:n_of_terms, :]
    # Handle nans:
    if seq.has(sp.nan):
        seq = seq[:list(seq).index(sp.nan), :]

    def an(till_now, x):
        coefs = x[:]
        coefs.reverse()
        out =  sp.Matrix([coefs[-1]*till_now.rows]) + till_now[-len(coefs[:-1]):, :].transpose()*sp.Matrix(coefs[:-1])
        return out
        # return (x[0] * till_now.rows + till_now[-len(x[1:]):, :].transpose() * x[1:, :])[0]

    reconst = seq[:max(len(x)-1, min(oeis_friendly, len(seq))), :]  # init reconst
    # else:
    #     reconst = seq[:len(x)-1, :]  # init reconst

    for i in range(len(seq) - len(reconst)):
        # reconst = reconst.col_join(sp.Matrix([an(reconst, x)]))
        reconst = reconst.col_join(an(reconst, x))

    out = reconst == seq, reconst, seq
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

    is_check = check_eq_man(x, id_, csv, oeis_friendly=34)
    print(is_check[0])
    print(list(is_check[1]))
    print(list(is_check[2]))
    print(list(is_check[1])[20:120])
    print(list(is_check[2])[20:120])


