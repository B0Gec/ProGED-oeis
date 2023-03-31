"""Run exact_ed on specific sequence with slurm arrays.
"""

import os
import sys
import time
import re

# import numpy as np  # not really since diophantine
import sympy as sp
import pandas as pd
import argparse

# needs (listed so far) doones,: exact_ed, diophantine_solver, linear_database_full.csv, buglist.py, oeil.sh, runoeil.sh


# if os.getcwd()[-11:] == 'ProGED_oeis':
#     from ProGED.examples.oeis.scraping.downloading.download import bfile2list
from exact_ed import exact_ed, timer, check_eq_man
# else:
#     from exact_ed import exact_ed, timer

# print("IDEA: max ORDER for GRAMMAR = floor(DATASET ROWS (LEN(SEQ)))/2)-1")

##############################
# Quick usage is with flags:
#  --seq_only=A000045 --sample_size=3 # (Fibonacci with 3 models fitted)
# search for flags with: flags_dict
###############


n_of_terms_load = 100000


VERBOSITY = 2  # dev scena
VERBOSITY = 1  # run scenario

DEBUG = True
DEBUG = False
BUGLIST = True
BUGLIST = False
if BUGLIST:
    from buglist import buglist

# if not DEBUG and BUGLIST:
#     print("\nWarning!!!!! buglist is used outside debug mode!!")
#     print("Warning!!!!! buglist is used outside debug mode!!")
#     print("Warning!!!!! buglist is used outside debug mode!!\n")

MAX_ORDER = 20  # We care only for recursive equations with max 20 terms or order.
N_OF_TERMS_ED = 200
TASK_ID = 0
JOB_ID = "000000"
SEQ_ID = (True, 'A153593')
# SEQ_ID = (False, 'A153593')
# EXPERIMENT_ID
timestamp = time.strftime("%Hh%Mm%Ss-%dd%m-%Y", time.localtime())
EXPERIMENT_ID = timestamp

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default=JOB_ID)
parser.add_argument("--task_id", type=int, default=TASK_ID)
parser.add_argument("-ss", type=int, default=-1)
parser.add_argument("-to", type=int, default=-1)
parser.add_argument("--order", type=int, default=MAX_ORDER)
parser.add_argument("--paral", type=int, default=2)
parser.add_argument("--verb", type=int, default=VERBOSITY)
parser.add_argument("--n_of_terms", type=int, default=N_OF_TERMS_ED)
parser.add_argument("--exper_id", type=str, default=EXPERIMENT_ID)
args = parser.parse_args()

job_id = args.job_id
task_id = args.task_id


MAX_ORDER = args.order
PARALLEL = args.paral
VERBOSITY = args.verb
experiment_id = args.exper_id

N_OF_TERMS_ED = 2*MAX_ORDER  # Use only first n_of_terms_ed of the given sequence.


flags_dict = {argument.split("=")[0]: argument.split("=")[1]
              for argument in sys.argv[1:] if len(argument.split("=")) > 1}
# n_of_terms_ed = 50
# n_of_terms_ed = int(flags_dict.get("--n_of_terms", n_of_terms_ed))




# from proged times:
has_titles = 1
# csv = pd.read_csv('oeis_selection.csv')[has_titles:]
#
# # for linear database:
# # mabye slow:
now = time.perf_counter()
# # a bit faster maybe:
csvfilename = 'linear_database_full.csv'

# print(os.getcwd())
if os.getcwd()[-11:] == 'ProGED_oeis':
    # csvfilename = 'ProGED_oeis/examples/oeis/' + csvfilename
    # print(os.getcwd())
    pass
# except ValueError as error:
#     print(error.__repr__()[:1000], type(error))

csv = pd.read_csv(csvfilename, low_memory=False, nrows=0)
n_of_seqs = len(list(csv.columns))
out_of_buglist = False
if BUGLIST:
    out_of_buglist = task_id >= len(buglist)

if task_id >= n_of_seqs or out_of_buglist:
    print('task_id surpassed our list')
else:
    seq_id = list(csv.columns)[task_id] if not SEQ_ID[0] or not DEBUG else SEQ_ID[1]
    if BUGLIST:
        seq_id = buglist[task_id]

    csv = pd.read_csv(csvfilename, low_memory=False, usecols=[seq_id])[:n_of_terms_load]
    # nans are checked by every function separately since exact_ed needs also ground truth

    csv.head()
    # now = timer(now, 'loading csv')
    # csv = csv.astype('int64')
    # print("csv", csv)
    # csv = csv.astype('float')
    terms_count, seqs_count = csv.shape






    # Run eq. disco. on all oeis sequences:
    start = time.perf_counter()
    now = start



    # with open('relevant_seqs.txt', 'r') as file:  # Use.
    #     # file.read('Hi there!')
    #     text = file.read()
    #
    # saved_seqs = re.findall(r'A\d{6}', text)


    def print_results(results, verbosity=2):

        if verbosity>=1:
            print("\n\n\n -->> The results are the following:  <<-- \n\n\n")
        if verbosity>=2:
            for (seq_id, eq, truth, x, is_reconst, is_check) in results:
                # if eq == 'EXACT_ED ERROR':
                #
                # else:
                print(seq_id, ': ', eq)
                print('truth:    ', truth)
                print('\"Check against website ground truth\":    ', is_reconst)
                # print('\"Manual check if equation is correct\":    ', is_check[0])
                print('\"Manual check if equation is correct\":    ', is_check)
                # if not is_check[0]:
                #     print('\" recorst vs truth\":    ', is_check[1].row_join(sp.zeros(is_check[1].rows, 1)).row_join(is_check[2]), '\n')
                print()

        if verbosity>=1:
            print(f"\n\n\n -->> The summary of the results untill now ({len(results)} sequences) are the following:  <<-- \n\n\n")

            false_positives = [seq_id for seq_id, eq, truth, x, is_reconst, is_check in results if not is_check]
            non_ground_truth = [seq_id for seq_id, eq, truth, x, is_reconst, is_check in results if not is_reconst]
            no_discovery = [seq_id for seq_id, eq, truth, x, is_reconst, is_check in results if x==[]]
            # ids = [seq_id for seq_id, eq, truth in results if eq == 'EXACT_ED ERROR']


            print('Number of false positives in exact ED:', len(false_positives))
            if verbosity >= 2:
                print('False positives:', false_positives)
            else:
                print('first 20 False positives:', false_positives[:20])
            print('Number of different sequences:', len(non_ground_truth))
            if verbosity >= 2:
                print('Different sequences:', non_ground_truth)
            else:
                print('first 20 Different sequences:', non_ground_truth[:20])
            print('Number of sequences without any equation found:', len(no_discovery))
            if verbosity >= 2:
                print('Sequences without any equation found:', no_discovery)
            else:
                print('first 20 Sequences without any equation found:', no_discovery[:20])

            # print('Number of false positives in exact ED:', len(ids))
            # print('False positives:', ids)

            print("Running equation discovery for all oeis sequences, "
                  "with these settings:\n"
                  f"=>> number of terms in every sequence saved in csv = {terms_count}\n"
                  # f"=>> nof_eqs = {nof_eqs}\n"
                  f"=>> number of all considered sequences = {n_of_seqs}\n"
                  # f"=>> list of considered sequences = {selection}\n"
                  f"=>> max order = {MAX_ORDER}\n"
                  )
            timer(now=start, text=f"While total time consumed by now")
        return 0

    SUMMARY_FREQUENCY = 20
    SUMMARY_FREQUENCY = 1000  # cluster for real
    # SUMMARY_FREQUENCY = 5
    INCREASING_FREQS = [2**i for i in range(SUMMARY_FREQUENCY) if 2**i <= SUMMARY_FREQUENCY]



    results = []

    def doone(task_id: int, seq_id: str, linear: bool, now=now):
        if VERBOSITY>=2:
            print()
        # try:
        x, coeffs, eq, truth = exact_ed(seq_id, csv, VERBOSITY, MAX_ORDER,
                                        n_of_terms=N_OF_TERMS_ED, linear=True)

        # order = list(x[1:]).index(0,)
        nonzero_indices = [i for i in range(len(x[1:])) if (x[i] != 0)]
        if nonzero_indices == []:
            ed_coeffs = []
        elif x[0] != 0:
            ed_coeffs = "containing non-recursive n-term"
        else:
            # order = len(nonzeros) - 1
            order = nonzero_indices[-1]

            # ed_coeffs = [str(c) for c in x[1:] if c!=0]
            ed_coeffs = [str(c) for c in x[1:1 + order]]

        if VERBOSITY>=2:
            print('ed_coeffs:', ed_coeffs)
        # print('coeffs:', coeffs)
        is_reconst = coeffs == ed_coeffs
        is_check_verbose = check_eq_man(x, seq_id, csv, n_of_terms=10**5)
        is_check = is_check_verbose[0]
        # print(f"{is_reconst}!, reconstructed as in ground truth.")
        # print(f"{is_check}!, \"manually\" checked if the equation holds for all terms.")

        # except Exception as error:
        #     print(type(error), ':', error)
        #     eq, truth, x = 'EXACT_ED ERROR', '\n'*3 + 'EXACT_ED ERROR!!, no output' + '\n'*3
        #     eq, truth, x = exact_ed(seq_id, csv, VERBOSITY)

        # results += [(seq_id, eq, truth, x, is_reconst, is_check_verbose)]

        print()
        if VERBOSITY>=2:
            now = timer(now=now, text=f"Exact ED for {task_id+1}-th sequence of {n_of_seqs} in "
                                      f"experiment set with id {seq_id} for first "
                                      f"{N_OF_TERMS_ED} terms with max order {MAX_ORDER} "
                                      f"while double checking against first {len(csv[seq_id])-1} terms.")
        elif VERBOSITY >= 1:
            # refreshrate = 1100
            refreshrate = 1
            if task_id % refreshrate == 0:
                _, timing_print = timer(now=start, text=f"While total time consumed by now, scale:{task_id+1}/{n_of_seqs}, "
                                      f"seq_id:{seq_id}, order:{MAX_ORDER}")
        # if task_id % SUMMARY_FREQUENCY == 0:
        #     print_results(results, verbosity=1)
        # elif task_id in INCREASING_FREQS:
        #     print_results(results, verbosity=1)
        # # except Exception as RuntimeError
        return seq_id, eq, truth, x, is_reconst, is_check, timing_print


    seq_id, eq, truth, x, is_reconst, is_check, timing_print = \
        doone(task_id=task_id, seq_id=seq_id, linear=True)
    # results += [doone(task_id=task_id, seq_id=seq_id)]
    # results += [(seq_id, eq, truth, x, is_reconst, is_check)]

    output_string = ""
    output_string += timing_print
    output_string += f"\n\n{seq_id}: \n{eq}\n"
    output_string += f"truth: \n{truth}\n\n"
    output_string += f'{is_reconst}  -  checked against website ground truth.     \n'
    output_string += f'{is_check}  -  \"manual\" check if equation is correct.    \n'

    # timer(now=start)

    sep = os.path.sep
    out_dir_base = f"results{sep}"
    out_dir = out_dir_base + f"{experiment_id}{sep}{job_id}{sep}"
    if DEBUG:
        out_dir = f"results_debug"
        print(output_string)

    out_fname = out_dir + f"{task_id:0>5}_{seq_id}.txt"
    os.makedirs(out_dir, exist_ok=True)

    if not DEBUG and not experiment_id == timestamp:
        f = open(out_fname, 'w')
        f.write(output_string)
        f.close()
        print(seq_id, ' done and written!')
    print(seq_id, ' done!')




# print(xv, verbose_eq)
# print(verbose_eq[:, 1:][], xv)

def prt(matrix: sp.Matrix):
    print(matrix.__repr__())
    return
# print(verbose_eq)
# prt(verbose_eq)
# print('re', verbose_eq.__repr__())



# # ad-hoc loop-check
# csv = pd.read_csv(csvfilename, low_memory=False, usecols=[i for i in range(SCALE)])[:n_of_terms]
# id = "A000027"
# eq = exact_ed(id, csv)
# print('\n', id)
# print(eq)
# print(check_eq(eq[0], id, csv))
# print(check_eq(eq[0], id, csv, sp.floor(n_of_terms/2 -1)))
