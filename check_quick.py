""" A quick 'manual' check if equation is correct, witout solution reference"""

import pandas as pd
import numpy as np

from exact_ed import unnan
seq_id = 'A000002'
# seq_id = 'A000045'
seq_id = 'A000041'
seq_id = 'A000108'
seq_id = 'A000984'
seq_id = 'A001045'

print('CHANGE seq_id !!!!!!!')


csv_filename = 'linear_database_newbl.csv'
csv_filename = 'cores_test.csv'

# # # checkall:
from saved_new_bfile10000 import bseqs
seq_full = bseqs[seq_id][1:]
overflow_terms = 250
overflow_terms = 550
overflow_terms = 35


n_of_terms = 12000
# n_of_terms = 12
csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])[:n_of_terms]
seq = unnan(csv[seq_id])
seq = seq_full[:overflow_terms]
seq = seq_full
print(seq)
# 1/0


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

# # implicit partitions a41: failed? seems
def an(n, an_1):
    res = an_1 ** 3 - 13 * an_1 ** 2 * n + 35 * an_1 * n ** 2 - 17 * n ** 3 + 67 * an_1 ** 2 - 288 * an_1 * n + 185 * n ** 2 + 360 * an_1 - 294 * n - 36
#         a(n)^3 -13*a(n)^2*n +35*a(n)*n^2 -17*n^3 +67*a(n)^2 -288*a(n)*n +185*n^2 +360*a(n) -294*n -36
    return res
a_zero = seq[0:1]

# catalan
def an(n, an, an_1):
    res = n*an -4*n*an_1 +6*an_1
    print(n, seq[n], seq[n-1], res)
    return res
a_zero = seq[0:1]

# a984:
def an(n, an, an_1, an_2):
# def an(n, an_1, an_2):
#     print('res = an * an_1 - 2 * an_1 ** 2 - 6 * an * an_2 + 16 * an_1 * an_2')
    res = an * an_1 - 2 * an_1 ** 2 - 6 * an * an_2 + 16 * an_1 * an_2
    res =  an_1 - 6  * an_2
    # # an * ( an_1 + 6  * an_2) = 2 * an_1 * ( an_1 - 8 * an_2)
    res = int(2 * an_1*(an_1 - 8 * an_2)/(an_1 - 6 * an_2))
    # a(n)  = 2 * a(n - 1) * (a(n - 1) - 8 * a(n - 2))/ * (a(n - 1) - 6 * a(n - 2))
# print(n, seq[n], seq[n-1], seq[n-2], res)
    # return res
    return res
a_zero = seq[0:2]

# a1045:
def an(n, an, an_1):
    res = an**2 -4*an*an_1 +4*an_1**2 -1
    # print(n, an, an_1, res)
    return res
a_zero = seq[0:2]


a = a_zero
def f_implicit(seq, a_zero, an=an):
    for n in range(len(a_zero), len(seq)):
        # a.append(an(n, seq[n-1], seq[n-2], seq[n-3], seq[n-4],))
        # a.append(an(n, seq[n]))
        a.append(an(n, seq[n], seq[n-1]))
        # print(an(n, seq[n], seq[n-1]))
        # print(n, seq[n], seq[n-1], a )
    return a


def reconstruct(seq, a_zero, an=an):
    for n in range(len(a_zero), len(seq)):
        # a.append(an(n, a[-1], a[-2], a[-3]))
        a.append(an(n, 1, a[-1], a[-2]))
    return a

azero_len = len(a_zero)
print(azero_len)

print(list(seq))
# a = reconstruct(seq, a_zero)
# print(a)
# print(a)
a = f_implicit(seq, a_zero)
print(a)
print('\ndiffs')

# print([i == a[n] for n,i in enumerate(seq)].index(False))
# print([i - a[n] for n,i in enumerate(seq)])
# print((np.array([i - a[n] for n,i in enumerate(seq)]) == 0).all())
# print((np.array([i for i in a[azero_len:]]) == 0))
print((np.array([i for i in a[azero_len:]]) == 0).all())

print(len(seq))

