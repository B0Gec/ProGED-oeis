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

# needs (listed so far) doones,: exact_ed, diophantine_solver, linear_database_full.csv


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
# n_of_terms_load = 100
# n_of_terms_load = 60
# n_of_terms_load = 30
# n_of_terms_load = 27
# n_of_terms_load = 10

SCALE = 1000
# SCALE = 10
# SCALE = 2

# SCALE = 40
# SCALE = 50
# SCALE = 100

SCALE = 50000

VERBOSITY = 2  # dev scena
VERBOSITY = 1  # run scenario


MAX_ORDER = 20  # We care only for recursive equations with max 20 terms or order.
N_OF_TERMS_ED = 200
TASK_ID = 0
JOB_ID = "000000"

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default=JOB_ID)
parser.add_argument("--task_id", type=int, default=TASK_ID)
parser.add_argument("--scale", type=int, default=SCALE)
parser.add_argument("-ss", type=int, default=-1)
parser.add_argument("-to", type=int, default=-1)
parser.add_argument("--order", type=int, default=MAX_ORDER)
parser.add_argument("--paral", type=int, default=2)
parser.add_argument("--verb", type=int, default=VERBOSITY)
parser.add_argument("--n_of_terms", type=int, default=N_OF_TERMS_ED)
parser.add_argument("--sub", type=bool, default=False)
args = parser.parse_args()

job_id = args.job_id
task_id = args.task_id


MAX_ORDER = args.order
PARALLEL = args.paral
VERBOSITY = args.verb
SUBMITIT = args.sub

SCALE = args.scale
# print(args.to, type(args.to))
RANGE = (args.ss, args.to) if args.to != -1 else (0, SCALE)

N_OF_TERMS_ED = 2*MAX_ORDER  # Use only first n_of_terms_ed of the given sequence.



flags_dict = {argument.split("=")[0]: argument.split("=")[1]
              for argument in sys.argv[1:] if len(argument.split("=")) > 1}
# n_of_terms_ed = 50
# n_of_terms_ed = int(flags_dict.get("--n_of_terms", n_of_terms_ed))
# SCALE = int(flags_dict.get("--scale", SCALE))
# SCALE = min(SCALE, )

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())



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
# try:
    # csv = pd.read_csv(csvfilename, low_memory=False, usecols=[i for i in range(SCALE)])[:n_of_terms_load]
# except ValueError as error:
#     print(error.__repr__()[:1000], type(error))
csv = pd.read_csv(csvfilename, low_memory=False, nrows=0)
n_of_seqs = len(list(csv.columns))
if task_id >= n_of_seqs:
    print('task_id surpassed our list')
else:
    seq_id = list(csv.columns)[task_id]

    csv = pd.read_csv(csvfilename, low_memory=False, usecols=[seq_id])[:n_of_terms_load]


    csv.head()
    # print(csv.head())
    # 1/0
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
    # CATALAN = "A000108"

    # pickle.dump(eq_discos, open( "exact_models.p", "wb" ) )

    #
    # selection_small = (
    #         "A000009",
    #         "A000040",
    #         "A000045",
    #         "A000124",
    #         # "A000108",
    #         "A000219",
    #         "A000292",
    #         "A000720",
    #         "A001045",
    #         "A001097",
    #         "A001481",
    #         "A001615",
    #         "A002572",
    #         "A005230",
    #         "A027642",
    #         )
    #
    selection = None
    # first seq id: A000004
    # last  seq id: A357130

    selection2 = (
            # "A000045",
            # "A000124",
            # "A000565",  # Nan at 25 term
            # "A016812",  # index error
            # "A000292",
            # "A001045",
            "A000032",
            "A021092",
            )
    # selection = selection2
    # selection = list(csv.columns)[:SCALE] if selection is None else selection
    selection = list(csv.columns)[RANGE[0]:RANGE[1]] if selection is None else selection
    # print(selection)
    # 1/0

    # with open('relevant_seqs.txt', 'r') as file:  # Use.
    #     # file.read('Hi there!')
    #     text = file.read()
    #
    # saved_seqs = re.findall(r'A\d{6}', text)
    # UNCOMMENT:
    # selection = saved_seqs if SCALE > 10100 else selection
    selection = selection

    # selection = selection_small
    # selection = saved_seqs[:5]
    # print(saved_seqs)
    # print(len(saved_seqs))

    output_string = ""
    # output_string = ("Running equation discovery for all oeis sequences, ",
    #     "with these settings:\n",
    #     f"=>> number of terms in every sequence saved in csv = {terms_count-1}\n",
    #     # f"=>> nof_eqs = {nof_eqs}\n",
    #     f"=>> number of all considered sequences = {n_of_seqs}\n",
    #     # f"=>> list of considered sequences = {selection}\n",
    #     )
    # output_string = "".join(output_string)


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

    def doone(task_id, seq_id, now=now):
        if VERBOSITY>=2:
            print()
        # try:
        x, coeffs, eq, truth = exact_ed(seq_id, csv, VERBOSITY, MAX_ORDER, n_of_terms=N_OF_TERMS_ED)

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

    # if SUBMITIT:
    #     import submitit
    #     print('\n\n ==== = = = = ======================\n going with SUBMITIT\n\n')
    #     log_folder = "log_submitit/%j"
    #     # a = [1, 2, 3, 4]
    #     # b = [10, 20, 30, 40]
    #     ns = [n for n in range(len(selection))]
    #     seq_ids = [selection[n] for n in ns]
    #
    #     executor = submitit.AutoExecutor(folder=log_folder)
    #     # the following line tells the scheduler to only run\
    #     # at most 2 jobs at once. By default, this is several hundreds
    #     #executor.update_parameters(slurm_array_parallelism=PARALLEL, slurm_partition="long")
    #     # executor.update_parameters(slurm_array_parallelism=PARALLEL)
    # #
    # #     parameters = {'slurm_array_parallelism': PARALLEL,
    # #                   'timeout_min': 1100,
    # #                   'slurm_partition':,
    # #     }
    # #     params = {key:parameters[key] for key in parameters if parameters[key] is not None}
    # #
    # #     if slurm_partition is not none:
    # #                   'slurm_partition':
    # #                   }
    #     executor.update_parameters(slurm_array_parallelism=PARALLEL, timeout_min=1010)
    #
    #     PARTED = False
    #     partsize = PARALLEL
    #     if PARTED:
    #         for i in ns:
    #             if i % partsize == 0:
    #                 jobs = executor.map_array(doone, ns[i*partsize:(i+1)*partsize], seq_ids[i*partsize:(i+1)*partsize])  # just a list of jobs
    #                 results += [job.result() for job in jobs]
    #
    #     else:
    #         jobs = executor.map_array(doone, ns, seq_ids)  # just a list of jobs
    #         results = [job.result() for job in jobs]
    #
    # else:
    # for n, seq_id in enumerate(selection):

    seq_id, eq, truth, x, is_reconst, is_check, timing_print = doone(task_id=task_id, seq_id=seq_id)
    # results += [doone(task_id=task_id, seq_id=seq_id)]
    # results += [(seq_id, eq, truth, x, is_reconst, is_check)]

    output_string += timing_print
    output_string += f"\n\n{seq_id}: \n{eq}\n"
    output_string += f"truth: \n{truth}\n\n"
    output_string += f'{is_reconst}  -  checked against website ground truth.     \n'
    output_string += f'{is_check}  -  \"manual\" check if equation is correct.    \n'

    DEBUG = True
    DEBUG = False
    # timer(now=start)

    sep = os.path.sep
    out_dir_base = f"results_oeis{sep}"
    out_dir = out_dir_base + f"{job_id}{sep}"
    if DEBUG:
        out_dir = f"results_debug"

    out_fname = out_dir + f"{task_id:0>5}_{seq_id}.txt"
    os.makedirs(out_dir, exist_ok=True)

    f = open(out_fname, 'w')
    f.write(output_string)
    # print(output_string)
    f.close()
    print(seq_id, ' done and written!')




# print(xv, verbose_eq)
# print(verbose_eq[:, 1:][], xv)

def prt(matrix: sp.Matrix):
    print(matrix.__repr__())
    return
# print(verbose_eq)
# prt(verbose_eq)
# print('re', verbose_eq.__repr__())



# # ad-hoc loop-check
# SCALE = 500
# csv = pd.read_csv(csvfilename, low_memory=False, usecols=[i for i in range(SCALE)])[:n_of_terms]
# id = "A000027"
# eq = exact_ed(id, csv)
# print('\n', id)
# print(eq)
# print(check_eq(eq[0], id, csv))
# print(check_eq(eq[0], id, csv, sp.floor(n_of_terms/2 -1)))
