"""Discover equation(s) on OEIS sequence to discover direct, recursive or even direct-recursive equation.
"""

import sympy as sp
import pandas as pd
import time

from ProGED.diophantine_solver import diophantine_solve
# from scrap_lin import timer

# print("IDEA: max ORDER for GRAMMAR = floor(DATASET ROWS (LEN(SEQ)))/2)-1")

def timer(now, text=f"\nScraping all (chosen) OEIS sequences"):
    before, now = now, time.perf_counter()
    consumed = now - before
    print(text +
          f" took:\n {round(consumed, 1)} seconds,"
          f" i.e. {round(consumed / 60, 2)} minutes"
          f" or {round(consumed / 3600, 3)} hours.")
    return now



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
VERBOSITY = 1  # run scenario

def exact_ed(seq_id, csv, verbosity=VERBOSITY):
    # max_order = 25
    max_order = None
    seq = sp.Matrix(csv[seq_id])
    truth = '(-34,45,1,-35,8)'
    # consts[0][1:-1].split(',')
    # consts = re.findall(r'\([-,\d]+\)', a.parent.text)
    # consts = re.findall(r'\([-,\d]+\)', b)
    # print(consts)
    # consts

    max_order = sp.floor(seq.rows/2)-1 if max_order is None else max_order
    data = grid_sympy(seq, max_order)

    m_limit = 3003
    b = data[max_order:(max_order + m_limit), 0]
    # b = max_order + m_limit
    # 1/0
    # 1/0
    A = data[max_order:(max_order + m_limit), 1:]
    # A = sp.Matrix(
    #     [[3, 0],
    #      [0, 3],
    #      [1, 0]])
    # b = sp.Matrix([6, 9, 2])

    if verbosity >= 2:
        print('A, b', A.__repr__(), b.__repr__())
        print('A[:4][:4] :', A[:6, :6].__repr__(), '\n', A[:, -2].__repr__())
        print('A, b  shapes', A.shape, b.shape)

    x = diophantine_solve(A, b)
    if verbosity >= 2:
        print('x', x)
    verbose_eq = ['a(n)', 'n']

    # print('max_order', max_order)
    for i in range(max_order):
        verbose_eq += [f"a(n-{i+1})"]
    verbose_eq = sp.Matrix([verbose_eq])
    # xv = np.array(x).astype(np.float64)
    # veq = verbose_eq[:, 1:]
    # print('types', type(x), type(veq))
    # , x.shape, veq.shape)

    if x==[]:
        print('NO EQS FOUND!!!')
        # 1/0
    else:
        if verbosity >= 1:
            print('We found an equation!!!:')
        x = x[0]
        expr = verbose_eq[:, 1:] * x
        eq = f"{verbose_eq[0]} = {expr[0]}"
        if verbosity >= 1:
            print('eq: ', eq)
        x = eq
    return x, truth


if __name__ == '__main__':
    # from proged times:
    has_titles = 1
    csv = pd.read_csv('oeis_selection.csv')[has_titles:]

    eq = exact_ed("A000045", csv)