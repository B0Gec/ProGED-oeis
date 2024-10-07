""" A quick 'manual' check if equation is correct, witout solution reference"""

import pandas as pd
import numpy as np

from exact_ed import unnan
seq_id = 'A000002'
# seq_id = 'A000045'


csv_filename = 'linear_database_newbl.csv'
csv_filename = 'cores_test.csv'

# # checkall:
# from saved_new_bfile10000 import bseqs
# seq_full = bseqs[seq_id][1:]


n_of_terms = 12000
csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])[:n_of_terms]
seq = unnan(csv[seq_id])
# seq = seq_full


a_zero = seq[0:4]
# # fibo:
# def an(n, an_1, an_2):
#     return an_1 + an_2

#  kolakoski:
def an(n, an_1, an_2, an_3):
    # res = (-an_1 * an_2 - 3*(an_1 + an_2) -7)/(an_1 + an_2 - 3)
    res = (-an_1 * an_2 - 3*(an_1 + an_2) -7)/(an_1 + an_2 - 3)
    res = an_1 * an_2 + an_1 * an_3 + an_2 * an_3 - 3* an_1 - 3* an_2 - 3* an_3 +7
    return res

# implicit kolakoski:
def an(n, an_1, an_2, an_3):
    # res = (-an_1 * an_2 - 3*(an_1 + an_2) -7)/(an_1 + an_2 - 3)
    res = (an_1 + an_2 - 3)  # often becomes zero => usless for calculation 1, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, -1, 0
    # res = an_1 * an_2 + an_1 * an_3 + an_2 * an_3 - 3* an_1 - 3* an_2 - 3* an_3 +7  # holds always
    return res

# implicit kolakoski:
def an(n, an_1, an_2, an_3, an_4):
    res = (an_1 + an_2 - 3)  # often becomes zero => usless for calculation 1, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, -1, 0
    res = an_1 * an_2 + an_1 * an_3 - an_2 * an_4 - an_3 * an_4 - 3 * an_1 + 3 * an_4  # holds for 10**4 terms
    # a(n) * ( a(n-1) + a(n-2) - 3) = a(n-1) * a(n-3) + a(n-2) * a(n-3) + 3 * a(n-3)
    # a_n * a_n_1 + a_n * a_n_2 - a_n_1 * a_n_3 - a_n_2 * a_n_3 - 3 * a_n + 3 * a_n_3,
    return res

a = a_zero
def f_implicit(seq, a_zero, an=an):
    for n in range(len(a_zero), len(seq)):
        a.append(an(n, seq[n-1], seq[n-2], seq[n-3], seq[n-4],))
    return a


a = a_zero
def reconstruct(seq, a_zero, an=an):
    for n in range(len(a_zero), len(seq)):
        a.append(an(n, a[-1], a[-2], a[-3]))
    return a


print(list(seq))
# a = reconstruct(seq, a_zero)
a = f_implicit(seq, a_zero)
print(a)
# print([i == a[n] for n,i in enumerate(seq)].index(False))
# print((np.array([i - a[n] for n,i in enumerate(seq)]) == 0).all())
print((np.array([a[n] for n,i in enumerate(seq[len(a_zero):])]) == 0).all())

print(len(seq))

