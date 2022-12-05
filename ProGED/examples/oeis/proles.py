"""Run equation discovery on OEIS sequences to discover direct, recursive or even direct-recursive equations.
"""

# import numpy as np
import sympy as sp
import pandas as pd
import time
# import sys
# import re

from exact_ed import exact_ed, timer

# print("IDEA: max ORDER for GRAMMAR = floor(DATASET ROWS (LEN(SEQ)))/2)-1")

##############################
# Quick usage is with flags:
#  --seq_only=A000045 --sample_size=3 # (Fibonacci with 3 models fited)  
# search for flags with: flags_dict
###############

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

SCALE = 1000
SCALE = 10
# SCALE = 100

# from proged times:
has_titles = 1
csv = pd.read_csv('oeis_selection.csv')[has_titles:]
#
# # for linear database:
# # mabye slow:
now = time.perf_counter()
# # a bit faster maybe:
csv = pd.read_csv('linear_database_full.csv', low_memory=False, usecols=[i for i in range(SCALE)])
csv.head()
print(csv.shape)
1/0
# now = timer(now, 'loading csv')
# # print(csv.)
# #
# csv = pd.read_csv('linear_database_full.csv', low_memory=False)
# csv.head()
# timer(now, 'loading csv')
# 1/0
#

# csv = csv.astype('int64')
# print("csv", csv)
# csv = csv.astype('float')
# print("csv", csv)
# 1/0
terms_count, seqs_count = csv.shape

# Old for fibonacci only:
# seq_id = "A000045"
# prt_id = "A000041"
# fibs = list(csv[seq_id])  # fibonacci = A000045
# prts = list(csv[prt_id])  # fibonacci = A000045
# # print("fibs", fibs)
# # fibs = np.array(fibs)
# # prts = np.array(prts)
# # oeis = fibs
# # sp_seq = sp.Matrix(csv[seq_id])
# # print(sp_seq)



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


# Run eq. disco. on all oeis sequences:

start = time.perf_counter()
now = start

# FIRST_ID = "A000000"
# LAST_ID = "A246655"
# # last_run = "A002378"
#
# start_id = FIRST_ID
# # start_id = "A000045"
# end_id = LAST_ID
# # end_id = "A000045"
#
# # start_id = "A000041"
# # end_id = "A000041"

CATALAN = "A000108"

# pickle.dump(eq_discos, open( "exact_models.p", "wb" ) )


selection = (
        "A000009",
        "A000040",
        "A000045",
        "A000124",
        # "A000108",
        "A000219",
        "A000292",
        "A000720",
        "A001045",
        "A001097",
        "A001481",
        "A001615",
        "A002572",
        "A005230",
        "A027642",
        )

selection2 = (
        "A000045",
        "A000124",
        # "A000292",
        # "A001045",
        )
selection = selection2


print("Running equation discovery for all oeis sequences, "
        "with these settings:\n"
        f"=>> number of terms in every sequence saved in csv = {terms_count}\n"
        # f"=>> nof_eqs = {nof_eqs}\n"
        f"=>> number of all considered sequences = {len(selection)}\n"
        f"=>> list of considered sequences = {selection}\n"
        )

VERBOSITY = 2  # dev scena
VERBOSITY = 1  # run scenario

results = []
for seq_id in selection:
    print()
    eq = exact_ed(seq_id, csv, VERBOSITY)
    results += [(seq_id, eq)]
    now = timer(now=now, text=f"Exact ED for {seq_id}")
    timer(now=start, text=f"While total time consumed by now")

# timer(now=start)
print("\n\n\n -->> The results are the following:  <<-- \n\n\n")
for (seq_id, eq) in results:
    print(seq_id, ': ', eq)


# print(xv, verbose_eq)
# print(verbose_eq[:, 1:][], xv)

def prt(matrix: sp.Matrix):
    print(matrix.__repr__())
    return
# print(verbose_eq)
# prt(verbose_eq)
# print('re', verbose_eq.__repr__())



