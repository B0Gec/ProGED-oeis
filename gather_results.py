"""
Gathers results from running oeis on cluster.
results/good/01234567
    - 34500_A000032.txt
    - ...

todo categories:

    - successfully found equations that are identical to the recursive equations written in OEIS (hereinafter - OEIS equation)
    - successfully found equations that are more complex than the OEIS equation ... 'manually' checked but ..!hey this equals next to last item in this list!
    - successfully found equations that do not apply do not apply to test cases ... 'manually' unchecked wrong (eq. holds just for first few terms).
    - successfully found equations that are valid for test cases but not valid in OEIS ... aka. 'manually' checked, but not OEIS equations
    - failure, no equation found. ... aka. We didn't found equations!


# all = failure + success
# success = manually fail + checked
# checked = unid + oeis
all = oeis + checked + manual fail + failure

ALSO!:
    - create false_truth list for blacklist.py:
        copy list false_non_man for experiment job_id=blacklist76
"""

##
import os
import re
import random
import string

import pandas as pd

from blacklist import no_truth, false_truth
false_truth_list = false_truth

# fname = 'results/good/01234567/34500_A000032.txt'
base_dir = "results/good/"
# job_id = "01234567"
# job_id = "36765084"  # old format
job_id = "36781342"  # 17.3. waiting 2 jobs  # to check
# job_id = "37100077"  # small
job_id = "37117747"
# job_id = "37117747_2"  # 23k succ
# job_id = "37256280"  # little of results
# job_id = "37256396"   # old school
# job_id = "37507488"   # old school
# job_id = "37507488_1"   # false_truth in progress
# job_id = "37256396_6" # false_truth in progress
# job_id = "37680622"     # false_truth in progress
job_id = "37683622"  # seems like for false_truth list creation
# job_id = "32117747"
# job_id = "debug2"
# job_id = "38095692"
# job_id = "sindy512135"
job_id = "sindygrid519_1"
# job_id = "incdio76"
job_id = "blacklist76"
# # job_id = "incdio86"
# # job_id = "i86/fix"
# job_id = "incdio74"  # 2023-07-04  ... I think this will be in final results.
# # job_id = "sindyens75"  # 2023-07-04
# # job_id = "sindydeb3"  # 2023-07-06
# # # job_id = "sindydeb3-1"  #
# # # job_id = "sindydeb3-2"  #
# # job_id = "sindydeb3-3"  #
# # job_id = "sindydeb3-4"  #
# # job_id = "sindydeb3-5"  #
# # job_id = "sindydeb3-7"  # 2023-07-10
# # job_id = "sindymerged"  #
#
# # # job_id = "diocores77"
# # # job_id = "diocor-merge"  #
# # job_id = "sindycore83"
# # # job_id = "dicor-cub"
# # # job_id = "dicor-cub19"
# # # job_id = 'dicor-ncub19_95'
# # # job_id = 'dicor-alibs96'  # bug since {eq}\nby lib{}\ntruth instead of {ed}\ntruth
# # # job_id = 'dicor-ncub10'  # max_order=10 only lib ncub  # bug since {eq}\nby lib{}\ntruth instead of {ed}\ntruth
# # # job_id = 'dicor-alib10'  # max_order=10 allibs
# # # job_id = 'sicor-ncub'
# # # job_id = 'sicor-ncub2'
# # # # job_id = 'sicor-nquad'
# # # # job_id = 'sicor-quad'
# # # # job_id = 'sicor-n'
# # job_id = 'sicor-lin'
# # # job_id = 'sicor-lin2'
# # # job_id = 'dicor-comb'
# # # job_id = 'sicor-comb'
# # # job_id = 'sicor-combalibs'
job_id = 'sicor-lin3'

# # # new "final?" results:
# # job_id = 'fdiocores'
# # job_id = 'fdiocorefix'
# job_id = 'fdiocorefix2'
#
# job_id = 'fdio'
# job_id = 'fdiobl'
# job_id = 'fdionewbl'
# # job_id = 'fdiocorr'
job_id = 'dicorrep'


# job_id = 'blacklist'  # 27237 non_ids

#
# # # job_id = 'sicor116'
# # # job_id = 'sicor9'
job_id = 'sicor9fix2'
# #
job_id = 'silin'
#
# job_id = 'sicor1114'

# job_id = 'dilin'
# job_id = 'findicor'
#
#
# # job_id = 'sideflin'  # fail: not even sindy
# # job_id = 'sidefcor'  # fail: not even sindy

# job_id = 'sdlin'
# # job_id = 'sdcor'  # fail: not core
# job_id = 'sdcor2'

print(job_id)

CORES = True if job_id in ("diocores77", 'diocor-merge', 'sindycore83', 'dicor-cub', 'dicor-cub19',
                           'fdiocores', 'fdiocorefix', 'fdiocorefix2', 'sicor116', 'dicorrep', 'sicor9fix2', 'sicor1114',
                           'findicor', 'sdcor2') else False
# CORES = True
# CORES = False
if CORES:
    csv_cols = list(pd.read_csv('cores_test.csv').columns)
    # print(csv_cols)
    # 1/0

time_complexity_dict = {
    'incdio74': '2h 30min (+ 20h+ for 9 sequences)',
    'sindyens75': '? ... still running',
    'sindydeb3': ' batch 1: 6.7.2023 from: 10.10 to 15:25 (5h 15min) ... still running',
    'sindydeb3-1': 'batch 2: 6.7. from 14.30 to 17:51 (3h 20min) ... still running',
    'sindydeb3-2': 'batch 3: 7.7. from 09.45 to 11:53 (2h 10min) ... still running',
    'sindydeb3-3': 'batch 4: 7.7. from 11:37 to 11:37 (0mins) ... still running',
    'sindydeb3-4': 'batch 5: 7.7. from 11:47 to 23:26 (11h 45min) ... still running',
    'sindymerged': '? ... still running',
    'diocores77': '25 mins',
    'diocor-merge': '25 mins',
    'sindycore83': '43 + mins',
    'dicor-cub': 'from 2pm..14.20, i.e. 20mins',
    'dicor-cub19': 'from 14.30 .. to 14:33, i.e. 20mins',
    'dicor-ncub19_95': 'from 5.9. 17h .. to more than 6.9. 8:25 ... to ?, i.e. >half day',
    'dicor-alibs96': '1day?',
    'dicor-ncub10': 'from 9:24  to 9:58 (147 files (34 non_ids)) (2023-9-7) ... to ? for all files',
    'dicor-alib10': 'from 9:56 (2023-9-7) to 10:30 or lets say 11:27 (excluding 11 jobs (known for bugs?)',
    'sicor-ncub': 'from 17:05 (2023-9-7) to ?',
    'sicor-nquad': 'from 17:05 (2023-9-7) to ?',
    'sicor-quad': 'from 17:20 (2023-9-7) to ?',
    'sicor-n': 'from 17:20 (2023-9-7) to ?',
    'sicor-lin': 'from 17:20 (2023-9-7) to ?',
    'sicor-lin2': 'from 17:30 (2023-9-7) to ?',
    'dicor-comb': 'from 18:27 (2023-9-11) to 18:28 (first file, first 33 files to 18:42, i.e. 15 mins) to 20:31 (34th file) to 13:35 (12.9)',
    'sicor-comb': 'from 15:42 (2023-9-13) to 15:42 (first file, first 33 files to ?, i.e. ? mins) to ',
    'sicor-combalibs': 'from 10:40? (2023-9-15) to 11:15 (first file, first 22 files to 11:22, i.e. ? mins) to ',
    'sicor-lin3': 'legit from 9:31 (2023-9-19) to 9:42 (29 reconstructions 22 files to ?, i.e. ? mins) to 9:56',
}
time_complexity_dict[job_id] = 'unknomn' if job_id not in time_complexity_dict else time_complexity_dict[job_id]
time_complexity = time_complexity_dict[job_id]


# seq_file = '13000_A079034.txt'
job_dir = base_dir + job_id + '/'
# fname = job_dir + seq_file
# print(os.listdir(exact_dir))

##
black_check = True

content_debug = """
While total time consumed by now, scale:371/34371, seq_id:A002278, order:20 took:
 1.5 seconds, i.e. 0.02 minutes or 0.0 hours.

A002278: 
a(n) = -10*a(n - 4) + a(n - 3) + 10*a(n - 1)
truth: 
a(n) = 11*a(n - 1) + -10*a(n - 2),  
a(0) = 0, a(1) = 4

False  -  checked against website ground truth.     
True  -  "manual" check if equation is correct.    
"""



VERBOSITY = 0
# VERBOSITY = 1
# VERBOSITY = 2
def extract_file(fname, verbosity=VERBOSITY, job_id=job_id):

    if verbosity >= 1:
        print()
        print(f'extracting file {fname}')
    f = open(fname, 'r')
    content = f.read()
    # output_string = content
    if verbosity >= 2:
        print(content)


    if job_id in ('dicor-ncub10', 'dicor-alibs96', ):
        eq = re.findall(r"A\d+:.*\n(.+)\nby library:", content)  # uncomment only for alibs96 and ncub10
    else:
        eq = re.findall(r"A\d+:.*\n(.+)\ntruth:", content)
    re_all_stars = re.findall(r"scale:\d+/(\d+),", content)
    re_found = re.findall(r"NOT RECONSTRUCTED", content)
    eq = None if len(re_found) > 0 or len(eq) == 0 else eq[0]
    # avg = True
    avg = False
    added = 'avg ' if avg else ''
    re_reconst = re.findall(r"\n(\w{4,5}).+checked against website ground truth", content)
    re_reconst = re.findall(r"\n(\w{4,5}).+" + f"checked {added}against website ground truth", content)
    re_manual = re.findall(r"\n(\w{4,5}).+" + f"\"manual\" check {added}if equation is correct", content)

    avg_vs_best = True
    avg_is_best = False
    if avg_vs_best:
        avg = True
        re_reconst_avg = re.findall(r"\n(\w{4,5}).+" + f"checked {added}against website ground truth", content)
        re_manual_avg = re.findall(r"\n(\w{4,5}).+" + f"\"manual\" check {added}if equation is correct", content)
        avg_is_best = re_reconst_avg == re_reconst and re_manual_avg == re_manual


    # print(len(re_all_stars))
    # print(content)
    if len(re_all_stars) == 0:
        pass
        # print('--pred')
        # print(content[:100])
        # print('--po')
    if verbosity >= 1:
        print('n_of_seqs', re_all_stars)
        print('refound', re_found)
        print('reconst', re_reconst)
        print('reman', re_manual)

    # n_of_seqs = int(re_all_stars[0])
    if len(re_all_stars) >= 1:
        n_of_seqs = int(re_all_stars[0])
    else:
        n_of_seqs = 0

    we_found = re_found == []
    if verbosity >= 1:
        print('we_found:', we_found)

    def truefalse(the_list):
        if black_check and the_list == []:
            return 0
        string = the_list[0]
        if string not in ('True', 'False'):
            if black_check:
                return 0
            print(string)
            raise ValueError
        else:
            return 1 if string == 'True' else 0
    is_reconst, is_check = tuple(map(truefalse, [re_reconst, re_manual]))

    PRINT_EQS = True
    if PRINT_EQS:
        if is_check and CORES:
            print('\'', is_check, fname[-11:-4], ':', eq, '\',')

    # re_manual =
    # we_found, is_reconst, is_check, = not (re_found == []), bool(re_reconst[0]), bool(re_manual[0]),

    # for now:
    # we_found = is_check

    # analysis: config vs success
    # re_configs = re.findall(r"\n(\w{4,5}).+" + f"checked {added}against website ground truth", content)
    re_configs = re.findall(r"\((\d{1,2}), (\d{1,2}), (\w{4,5}), (\w{4,5})\),", content)
    confs = map(lambda stri: (stri[:2], True if stri[2:] == ('True', 'True') else False), re_configs)
    # trueconfs = [conf[0] for conf in confs if conf[1]] # nonsense - no all true config!
    # print(list(trueconfs))
    confs = list(confs)

    # 1/0

    f.close()

    return we_found, is_reconst, is_check, n_of_seqs, avg_is_best, confs, eq

# print(os.getcwd())
# print(os.listdir())
# print('extract', extract_file(fname))

# def for_loop(dir: str):
#     for seq_file in os.listdir(dir):
#         # is = extract_file(seq_file)
#         # seq_id, eq, truth, x, is_reconst, is_check, timing_print = extract_file(seq_file)
#         # ts =
#         # all = oeis + checked + manual
#
#     return vars


from functools import reduce

# sez = [isfail,
#         ]
#
# all = fail + nonmanual + nonid + idoeis
#
# fail = not we found
# manually = is_checked
#
# we_found


dic = {'a': (True, False, False, True),
       'b': (False, False, True, False),
       'c': (False, True, False, True),
       }
l = ['b', 'c', 'a',]
debug = True

def for_summary(aggregated: tuple, fname: str):

    # now -> f, m, i, o
    we_found, is_reconst, is_checked, _, avg_is_best, trueconfs, eq = extract_file(job_dir + fname)

    id_oeis = is_reconst
    non_id = not is_reconst and is_checked
    non_manual = not is_checked and we_found
    fail = not we_found

    # makes no sense, e.g. is_reconst and not is_checked
    reconst_non_manual = is_reconst and not is_checked
    # non_manual = we_found and not is_checked


    buglist, job_bins, non_id_list, ed_fail_list, non_manual_list, agg_confs = aggregated[-6:]
    if debug:
        if reconst_non_manual:
            buglist += [fname]
            # print(aggregated, fname)
            # raise IndexError("Bug in code!")
        if black_check and non_manual:
            buglist += [fname]
        if non_id:
            non_id_list += [fname]
        if fail:
            ed_fail_list += [fname]
        if non_manual:
            non_manual_list += [fname]

    task_id = int(fname[:5])
    # Fail analysis:
    # a. 34 bins for jobs
    job_bins[task_id//1000] += 1
    # print(job_bins)
    # bins = [bin0, bin1, ... bin 34]

    # print(len(trueconfs))
    agg_confs = trueconfs if agg_confs == ['start'] else agg_confs
    # trueconfs = [(conf[0], conf[1]+trueconfs[n][]) for n, conf in enumerate(agg_confs)]
    new_confs = [(x[0], x[1]+y[1]) for x, y in zip(agg_confs, trueconfs)]
    # print(new_confs)
    # print(len(new_confs))



    # summand = [f, m, i, o]
    to_add = (id_oeis, non_id, non_manual, fail, reconst_non_manual, avg_is_best)


    # f, m, i, o = aggregated
    if VERBOSITY >= 1:
        print('to_add sum:', sum(to_add))
        print('to_add:', to_add)
        print('aggregated:', aggregated)
        if sum(to_add[:4]) - sum(to_add[4:]) > 1:
           print(' --- --- look here -s- --- ')
           raise ArithmeticError
        # if sum(to_add[:4]) > 1:
        #     print(reconst_non_manual)
        #     print(' --- --- look here --- --- ')
        #     raise ArithmeticError




    zipped = zip(aggregated[:-2], to_add)
    counters = tuple(map(lambda x: x[0] + x[1], zipped))
    return counters + (buglist, job_bins, non_id_list, ed_fail_list, non_manual_list, new_confs)

# # Hierarhical:
# files_subdir = [list(map(lambda x: f'{subdir}{os.sep}{x}',
#                          os.listdir(job_dir + subdir))) for subdir
#                 in os.listdir(job_dir)]
# flatten = sum(files_subdir, [])
# files = flatten
# one for all:
files = os.listdir(job_dir)



# # # # # debugging:
# # # # files = list(map(lambda file: file[9:14], files))
cut = (9, 14, 15, 22)
cut = tuple(i-9 for i in cut)
# success_ids_pairs = sorted(list(map(lambda file: (file[cut[0]:cut[1]], file[cut[2]:cut[3]]), files)))
# print(success_ids_pairs)
success_ids = list(map(lambda file: file[cut[2]:cut[3]], files))
start = 13000
start = 10000
start = 0
limited_runs = 123456
from all_ids import all_ids
if CORES:
    all_ids = csv_cols
all_ids_ref = all_ids
all_ids = all_ids[start:limited_runs]

# succsess_task_ids =  sorted([all_ids_ref.index(id_) for id_ in success_ids])
# print('task_ids of jobs successful:', succsess_task_ids)
# print(str(succsess_task_ids).replace(' ', '').strip('[]'))

# print('all_ids', all_ids[:10])
# renaming = [(job_dir + file, job_dir + all_ids.index(str(file[cut[2]:cut[3]])) + file[cut[0]:cut[1]]) for file in files]
# renaming = [(job_dir + file, job_dir + f"{all_ids.index(str(file[cut[2]:cut[3]])):0>5}_{file[cut[2]:cut[3]]}.txt") for file in files]
# list(map(lambda pair: os.rename(pair[0], pair[1]), renaming))
# print(renaming[:10])
# 1/0


# # print(all_ids[:10])
# # print(len(success_ids), len(all_ids)-len(success_ids))
# # # unsuccessful = [(f"{task:0>5}", id_) for task, id_ in enumerate(all_ids) if (f"{task:0>5}", id_) not in success_ids_pairs]
# # # print('first few unsuccessful:')
# # # print(unsuccessful[:10])
# # # print(len(success_ids_pairs), len(unsuccessful), len(success_ids_pairs) + len(unsuccessful), len(all_ids))


# a. check for not blacklisted unsuccessful jobs
from blacklist import blacklist, no_truth
## sanity check: successful_id in blacklist
# successful_black = [i for i in success_ids if i in blacklist]
# print('sanity check:', successful_black)
# print(sorted(blacklist[:10]), '\n', sorted(success_ids[:10]))

# print(len(blacklist), len(set(blacklist)))
# not_blacklisted = [(task, i) for task, i in unsuccessful if not i in blacklist]
# blacklisted = [(task, i) for task, i in enumerate(all_ids) if not i in blacklist]
not_blacklisted = [i for i in all_ids if not i in blacklist and not i in success_ids]
# not_blacklisted = [i for n, i in enumerate(all_ids) if not i in blacklist and not i in success_ids and i]
print('jobs failed (unsuccessful):', not_blacklisted[:10], len(not_blacklisted))
# not_missing_truth = [i for i in all_ids if not i in no_truth and not i in success_ids]
# not_blacklisted_pairs = [(f"{task:0>5}", id_) for task, id_ in enumerate(all_ids)
#                          if not id_ in blacklist and not id_ in success_ids]

# jobs_failed_per_bin = [0 for _ in range(35)]
failed_ids =  [all_ids_ref.index(id_) for id_ in not_blacklisted]
# print('ids of jobs failed (unsuccessful):', failed_ids)
# print(str(failed_ids).replace(' ', '').strip('[]'))
# print('ids of jobs failed (unsuccessful):', [all_ids_ref.index(id_) for id_ in not_blacklisted])
not_blacklisted_pairs = [(f"{all_ids_ref.index(id_):0>5}", id_) for id_ in not_blacklisted]
print(not_blacklisted_pairs[:10], len(not_blacklisted_pairs))
# print(not_blacklisted_pairs[:3100], len(not_blacklisted_pairs))

def failed_bins(agg, pair):
    agg[int(pair[0])//1000] += 1
    return agg

bins_failed = reduce(failed_bins, not_blacklisted_pairs, [0 for _ in range(35)])
for n, jobs in enumerate(bins_failed):
    print(n, jobs)

print(bins_failed)
print('here i am')
# 1/0
# ['A044941', 'A053833', 'A055649'] 3
# [('00184', 'A001299'), ('00185', 'A001300'), ('00186', 'A001301'), ('00187', 'A001302'), ('00195', 'A001313'), ('00196', 'A001314'), ('00198', 'A001319'), ('00222', 'A001492'), ('00347', 'A002015'), ('00769', 'A005813')] 1921
# these are blacklisted due to false ground truth
# print('A055649' in blacklist)
# 1/0


# # buglist = not_blacklisted
# # # output_string = f'successful_list = {successful_list}'
# # output_string = f'buglist = {buglist}'
# # # writo new files:
# # out_fname = 'buglist.py'
# # f = open(out_fname, 'w')
# # f.write(output_string)
# # f.close()




# from buglist import buglist
# unsucc_bugs = [i for i in buglist if not i in success_ids]
# indices = [all_ids.index(i) for i in unsucc_bugs]
#
# print('unsucc', indices)
# # print(buglist[:10], len(buglist))
# # unsucc [11221, 27122, 27123]


# 1/0
# [('00191', 'A001306'), ('00193', 'A001310'), ('00194', 'A001312'), ('00200', 'A001343'), ('00209', 'A001364'), ('00210', 'A001365'), ('00946', 'A007273'), ('01218', 'A008685'), ('01691', 'A011616'), ('01692', 'A011617')]



scale = 40
scale = 4000
scale = 50100
files_debug = files[0:scale]
files = files_debug
# print(files)

_a, _b, _, n_of_seqs, avg_is_best, true_confs, eq = extract_file(job_dir + files[0])
if CORES:
    n_of_seqs = 164
# print(n_of_seqs)

# # 10.) checked that no_truth == mia task ids from experiment job_id = "blacklist76"
# csv_filename = 'linear_database_full.csv'
# csv = pd.read_csv(csv_filename, low_memory=False, nrows=0)
# files_task_ids = [file[:5] for file in files]
# mia_task_ids = [task_id for task_id in range(n_of_seqs) if f"{task_id:0>5}" not in files_task_ids]
# mia_ids = [csv.columns[i] for i in mia_task_ids]
# print(mia_ids[0])
# # print(mia_task_ids, len(mia_task_ids))
# # print(mia_task_ids[:6], len(mia_task_ids))
# print(len(mia_task_ids), len(no_truth), mia_task_ids == no_truth, no_truth[0], mia_task_ids[0])
# print(len(mia_ids), len(no_truth), mia_ids == no_truth, no_truth[0], mia_ids[0])
# 1/0
#


# summary = reduce(for_summary, files, (0, 0, 0, 0, 0,))
# summary = reduce(for_summary, files[:], (0, 0, 0, 0, 0, 0, [], [0 for i in range(36)], [], [], [], ['start']))  # save all buggy ids
summary = reduce(for_summary, sorted(files[:]), (0, 0, 0, 0, 0, 0, [], [0 for i in range(36)], [], [], [], ['start']))  # save all buggy ids
# print(summary)
# 1/0

# corrected_sum = sum(summary[:4]) - sum(summary[4:])
corrected_sum = sum(summary[:4]) - sum(summary[4:5])
print(corrected_sum)
# 1/0
print()
print(str(summary)[:1*10**2])
print(f'all files:{len(files)}, sum:{sum(summary[:4])}, corrected sum: {corrected_sum}')
# print(f((1,2,3,4,), 'c'))

print()
print(f'Results: ')


no_truth, false_truth = len(no_truth), len(false_truth)
if CORES:
    no_truth, false_truth = 0, 0
ignored = no_truth + false_truth
print(ignored)

# all_seqs = 34371
n_of_seqs_db = n_of_seqs
print(n_of_seqs_db)
n_of_seqs = n_of_seqs - ignored
jobs_fail = n_of_seqs - len(files)  # or corrected_sum.
print(n_of_seqs)
print(jobs_fail)

id_oeis, non_id, non_manual, ed_fail, reconst_non_manual, avg_is_best, buglist, \
    job_bins, non_id_list, ed_fail_list, non_manual_list, trueconfs = summary
corrected_non_manual = non_manual - reconst_non_manual
all_fails = ed_fail + jobs_fail

official_success = id_oeis + non_id

# for latex new experiment variables:
forbidden = ['S', 'U', 'V', 'G', 'X', 'P', 'I']
my_alphabet = [i for i in string.ascii_uppercase if i not in forbidden]
# my_alphabet = ['G']
random_symbol = random.choice(my_alphabet)
symbol = random_symbol + random_symbol

printout = f"""
    {id_oeis: >5} = {id_oeis/n_of_seqs*100:0.3} % ... (id_oeis) ... successfully found equations that are identical to the recursive equations written in OEIS (hereinafter - OEIS equation)
    {non_id: >5} = {non_id/n_of_seqs*100:0.3} % ... (non_id) ... successfully found equations that are more complex than the OEIS equation 
    {non_manual: >5} = {non_manual/n_of_seqs*100:0.3} % ... (non_manual) ... successfully found equations that do not apply to test cases 
    {ed_fail: >5} = {ed_fail/n_of_seqs*100:0.3} % ... (fail) ... failure, no equation found. (but program finished smoothly, no runtime error)
    {reconst_non_manual: >5} = {reconst_non_manual/n_of_seqs*100:0.3} % ... (reconst_non_manual) ... fail in program, specifically: reconstructed oeis and wrong on test cases.
    ~~{corrected_non_manual:~>5} = {corrected_non_manual/n_of_seqs*100:0.3} % ... (corrected_non_manual = non_manual ... reconst_non_manual) ... non_manuals taking bug in my code into the account.~~

    {(id_oeis + non_id + corrected_non_manual + ed_fail): >5} ... (id_oeis + non_id + corrected_non_manual + fail) = sum ... of all files
    
    
    {no_truth: >5} = {no_truth/n_of_seqs_db*100:0.3} % ... sequences with no ground truth -> missing jobs
    {false_truth: >5} = {false_truth/n_of_seqs_db*100:0.3} % ... sequences with false ground truth -> missing jobs
    {ignored: >5} = {ignored/n_of_seqs_db*100:0.3} % ... sequences with no or false ground truth -> all missing jobs by default
    {n_of_seqs_db: >5} ... all sequences in our dataset
    
    {jobs_fail: >5} = {jobs_fail/n_of_seqs*100:0.3} % ... runtime errors = jobs failed
    {all_fails: >5} = {all_fails/n_of_seqs*100:0.3} % ... all fails  <--- (look this) ---
    {n_of_seqs: >5} ... all considered sequences, i.e. in our dataset - ignored
    
    _______________________________________________________
    {id_oeis + non_id + corrected_non_manual + all_fails} ... (id_oeis + non_id + corrected_non_manual + all_fails) ... under the line check if it sums up
    {id_oeis + non_id + corrected_non_manual + ed_fail + ignored} ... (id_oeis + non_id + corrected_non_manual + ed_fails + ignored) ... under the line check if it sums up
    {id_oeis + non_id + corrected_non_manual + ed_fail + ignored + jobs_fail} ... (id_oeis + non_id + corrected_non_manual + ed_fails + ignored + jobs_fail) ... under the line check if it sums up
    
    {official_success: >5} = {official_success/n_of_seqs*100:0.3} % - official success (id_oeis + non_id)
    
    {avg_is_best: >5} = avg is best ... I might be wrong ... 
    
    time complexity: {time_complexity}
    
    
    
    ==========================================================

    \\newcommand{{\\allSeqs}}{{{n_of_seqs}}}
    \\newcommand{{\\isOeis}}{{{id_oeis}}}
    \\newcommand{{\\nonId}}{{{non_id}}}
    \\newcommand{{\\nonMan}}{{{non_manual}}}
    \\newcommand{{\\edFail}}{{{ed_fail}}}
    \\newcommand{{\\jobFail}}{{{jobs_fail}}}

    \\newcommand{{\\{symbol}allSeqs}}{{{n_of_seqs}}}
    \\newcommand{{\\{symbol}isOeis}}{{{id_oeis}}}
    \\newcommand{{\\{symbol}nonId}}{{{non_id}}}
    \\newcommand{{\\{symbol}nonMan}}{{{non_manual}}}
    \\newcommand{{\\{symbol}edFail}}{{{ed_fail}}}
    \\newcommand{{\\{symbol}jobFail}}{{{jobs_fail}}}

    \\FPeval{{\\{symbol}ok}}{{           clip(\\{symbol}isOeis+\\{symbol}nonId)}}
    \\FPeval{{\\{symbol}notOeis}}{{      clip(\\{symbol}nonMan+\\{symbol}nonId)}}
    \\FPeval{{\\{symbol}notIsFound}}{{   clip(\\{symbol}edFail+\\{symbol}jobFail)}}
    \\FPeval{{\\{symbol}isFound}}{{      clip(\\{symbol}isOeis+\\{symbol}notOeis)}}
    \\FPeval{{\\{symbol}allSeqsc}}{{     clip(\\{symbol}isFound+\\{symbol}notIsFound)}}
    
    \\FPeval{{\\perc{symbol}IsOeis}}{{       round(100 * \\{symbol}isOeis/\\{symbol}allSeqs:2)}}
    \\FPeval{{\\perc{symbol}NonId}}{{        round(100 * \\{symbol}nonId/\\{symbol}allSeqs:2)}}
    \\FPeval{{\\perc{symbol}NonMan}}{{       round(100 * \\{symbol}nonMan/\\{symbol}allSeqs:2)}}
    \\FPeval{{\\perc{symbol}NotIsFound}}{{   round(100 * \\{symbol}notIsFound/\\{symbol}allSeqs:2)}}

    
        \\begin{{table}}[h]
    \\caption{{Table describing the {job_id} results of novel Diofantos method experiments so far.}}
    \\label{{tab:dio.cores}}
       \\renewcommand{{\\arraystretch}}{{1.5}} \\begin{{center}}
    \\begin{{tabular}}{{ll|cc|c}} \\toprule \\centering
    &  \\textbf{{outputed}} &  & \\textbf{{not\\_outputed}} &  \\\\
     \\textbf{{identical}} & \\textbf{{nonidentical}} & \\textbf{{error free (ed\\_fail)}} 
    & \\textbf{{runtime error (jobs\\_fail)}} & $\\Sigma$ \\\\
        \\midrule
        ? (\\{symbol}isOeis) & \\{symbol}nonId  &  \\{symbol}edFail & \\{symbol}jobFail &  \\{symbol}allSeqsc \\\\
        \\bottomrule \\end{{tabular}} \\end{{center}}
    \\end{{table}}

\\begin{{figure}}
    \\centering
\\Tree[.$\\{symbol}allSeqs$ [.outputed [.$\\{symbol}isOeis$ ] 
                    [.$\\{symbol}nonId$ 
                    ] ] 
               [.not\\_outputed 
                    [.$\\{symbol}edFail$ ]
                    [.$\\{symbol}jobFail$ ]
                    ]] 
    \\caption{{Tree explaining the process of grouping sequences into disjoint bins with tangible numbers for {job_id}.}}
    \\label{{fig:explain_numbers{symbol}}}
\\end{{figure}}

\\begin{{figure}}
    \\begin{{tikzpicture}}
    % \\pie[text=legend,
        \\pie[color={{ipsscblue, ipsscred, ipsscorange, ipsscyellow}}] 
        {{   \\perc{symbol}IsOeis/identical (\\{symbol}isOeis),
            \\perc{symbol}NonId/ nonidentical (\\{symbol}nonId),
            % \\perc{symbol}NonMan/invalid (\\{symbol}nonMan),
            \\perc{symbol}NotIsFound/no output (\\{symbol}notIsFound) 
            }}
    \\end{{tikzpicture}}

    \\caption{{Summary of (fresh) results with pie chart for {job_id}.}}
    \\label{{fig:explain_tree{symbol}}}
\\end{{figure}}

"""


print(printout)
# 1/0

n = 6
print(n)
print(f'first {n} bugs:', buglist[:n])
print(len(buglist))
print(f'job bins (task_id= 0, 1, ... 34):', job_bins)
print(f'zipped job bins (task_id= 0, 1, ... 34):', [(n, i) for n, i in enumerate(job_bins)])
print(f'check bins: {len(files)} = {sum(job_bins)} ?')
for n, i in enumerate(job_bins):
    print(n, i)

m = 165
print(f'first {m} non_ids:', non_id_list[:m])
non_id_task_ids = ','.join([str(int(fname[:5])) for fname in non_id_list])
print('task_ids of non_ids:', non_id_task_ids[:100], '...')
# print(str(succsess_task_ids).replace(' ', '').strip('[]'))

print(len(non_id_list))
n = 1700
print(f'first {n} non_manuals:', sorted(non_manual_list[:n]))
# 1/0

# print(f'all non_manuals:', non_manual_list)
# check if new false_truth blacklist contains all old false_truths:  # experiment job_id = "blacklist76"
false_non_man = [i[6:6+7] for i in non_manual_list]
# print(false_non_man)
# print(false_non_man[0], len(false_non_man), false_non_man[:6], false_truth_list[:6], len(false_truth_list))
# print('sanity', sorted([i for i in false_truth_list if i not in false_non_man]))
# print('new', [i for i in false_non_man if i not in false_truth_list])
# 1/0

# print(len(non_manual_list))
print(f'first {n} ed_fails:', ed_fail_list[:n])
print(len(ed_fail_list))
print(f'first {n} true configs:', trueconfs[:n])
print(len(trueconfs))
# generated by copilot with """print('intersection of non_ids and non_manua""" ...
# print('intersection of non_ids and non_manuals:', len(set(non_id_list) & set(non_manual_list)))
# print('intersection of non_manuals and ed_fail_list:', len(set(non_manual_list) & set(ed_fail_list)))
# print('intersection of non_ids and ed_fail_list:', len(set(non_id_list) & set(ed_fail_list)))

def fname2id(fname):
    return re.findall("A\d{6}", fname)[0]

bugids = list(map(fname2id, buglist))
# print(bugids)
# print(buglist)
# print('len(bugids)', len(bugids))
# write_bugs = True
write_blacklist = True
write_blacklist = False
write_bugs = False
# write_bugs = True

# # from blacklistit import no_truth
# from blacklist import blacklist_old, no_truth
# if write_blacklist:
#     output_string = ""
#     output_string += f'blacklist_old = {blacklist_old}\n'
#     output_string += f'no_truth = {no_truth}\n'
#     output_string += f'false_truth = {bugids}\n'
#
#     # writo new files:
#     out_fname = 'blacklist.py'  # v 18.4.2023
#     # f = open(out_fname, 'r')
#     # before = f.read()
#     print('output:')
#     # output_string = f'{before}\n{output_string}'
#     print(output_string)
#     # f.close()
#
#     # f = open(out_fname, 'w')
#     # f.write(output_string)
#     # f.close()

# if write_bugs:
#     f = open('non_manuals.py', 'w')
#     f.write(f'non_manuals = {non_manual_list}')
#     # f = open('ed_fails.py', 'w')
#     # f.write(f'ed_fails = {ed_fail_list}')
#     # f = open('buglist.py', 'w')
#     # f.write(f'buglist = {bugids}')
#     f.close()
#
# # test import:
# from buglist import buglist as bl
# from blacklist import blacklist_old, no_truth, false_truth
# print(len(blacklist_old), blacklist_old)
# print(len(no_truth))
# print(len(false_truth))

# first 6 bugs: ['37100080/01230_A053833.txt', '37100079/00095_A166986.txt', '37100079/00278_A055649.txt']
# 37256442/05365_A026471.txt', '37256442/05484_A027636.txt', '37256442/05778_A028253.txt', '37256442/05478_A027630.txt', '37256442/05367_A026474.txt', '37256442/05480_A027632.txt']

# compare:
ncub = ['00010_A000035.txt', '00009_A000032.txt', '00014_A000045.txt', '00017_A000058.txt', '00019_A000079.txt', '00030_A000129.txt', '00038_A000204.txt', '00041_A000225.txt', '00042_A000244.txt', '00048_A000302.txt', '00051_A000326.txt', '00057_A000583.txt', '00075_A001045.txt', '00088_A001333.txt', '00095_A001519.txt', '00100_A001906.txt', '00106_A002275.txt', '00111_A002530.txt', '00112_A002531.txt', '00114_A002620.txt', '00123_A004526.txt', '00130_A005408.txt', '00133_A005843.txt']
alibs = ['00009_A000032.txt', '00010_A000035.txt', '00014_A000045.txt', '00017_A000058.txt', '00019_A000079.txt',
   '00029_A000124.txt', '00030_A000129.txt', '00038_A000204.txt', '00039_A000217.txt', '00041_A000225.txt',
   '00042_A000244.txt', '00046_A000290.txt', '00047_A000292.txt', '00048_A000302.txt', '00051_A000326.txt',
   '00052_A000330.txt', '00056_A000578.txt', '00057_A000583.txt', '00061_A000612.txt', '00067_A000798.txt',
   '00075_A001045.txt', '00077_A001057.txt', '00088_A001333.txt', '00095_A001519.txt', '00097_A001699.txt',
   '00100_A001906.txt', '00106_A002275.txt', '00108_A002378.txt', '00111_A002530.txt', '00112_A002531.txt',
   '00114_A002620.txt', '00123_A004526.txt', '00130_A005408.txt', '00133_A005843.txt', '00158_A055512.txt']

ncub10 = ['00009_A000032.txt', '00010_A000035.txt', '00014_A000045.txt', '00017_A000058.txt', '00019_A000079.txt', '00029_A000124.txt', '00030_A000129.txt', '00038_A000204.txt', '00039_A000217.txt', '00041_A000225.txt', '00042_A000244.txt', '00046_A000290.txt', '00047_A000292.txt', '00048_A000302.txt', '00051_A000326.txt', '00052_A000330.txt', '00056_A000578.txt', '00057_A000583.txt', '00067_A000798.txt', '00075_A001045.txt', '00077_A001057.txt', '00088_A001333.txt', '00095_A001519.txt', '00097_A001699.txt', '00100_A001906.txt', '00106_A002275.txt', '00108_A002378.txt', '00111_A002530.txt', '00112_A002531.txt', '00114_A002620.txt', '00123_A004526.txt', '00130_A005408.txt', '00133_A005843.txt', '00158_A055512.txt']
ncub = alibs
ncub = ncub10

dicores = ['00133_A005843.txt', '00009_A000032.txt', '00088_A001333.txt', '00095_A001519.txt', '00046_A000290.txt', '00047_A000292.txt', '00052_A000330.txt', '00010_A000035.txt', '00042_A000244.txt', '00111_A002530.txt', '00057_A000583.txt', '00075_A001045.txt', '00108_A002378.txt', '00123_A004526.txt', '00038_A000204.txt', '00056_A000578.txt', '00041_A000225.txt', '00019_A000079.txt', '00130_A005408.txt', '00100_A001906.txt', '00106_A002275.txt', '00051_A000326.txt', '00014_A000045.txt', '00114_A002620.txt', '00077_A001057.txt', '00112_A002531.txt', '00030_A000129.txt']
dicores = alibs
print(sorted(ncub))
print(sorted(dicores))
print(len(ncub), len(dicores))
print(sorted(list(set(ncub) & set(dicores))))
print(sorted(list(set(ncub).symmetric_difference(set(dicores)))))
print(sorted(list(set(ncub).difference(set(dicores)))))
print(sorted(list(set(dicores).difference(set(ncub)))))


import numpy as np
# gt = pd.read_csv('gt1125.csv')
gt = pd.read_csv('ground_truth - ground_truth918_3.csv')
gt = pd.read_csv('ground_truth - ground_truth922.csv')
# gt = pd.read_csv('gt919.csv')
gt_sin = gt['SINDy']
# gt_sin = gt['Diofantos [disco., outputed]']
print(gt)
print(gt.columns)
print(gt_sin[112], gt[gt.columns[0]][112])
# 1/0

# print(gt_sin[1], type(gt_sin[1]), )
# print(gt_sin == np.nan)
cat_name = 'cathegory (trivial [T]/exists [E]/hard [H])'  # x, h, v
print('cat', gt[cat_name][1], type(gt[cat_name][1]))
seqid_name = 'Unnamed: 0'
discos = [(n, gt[seqid_name][n], i) for n, i in enumerate(gt_sin) if isinstance(i, str) and 'yes' in i]
teh = [(n, gt[cat_name][n], i) for n, i in enumerate(gt[cat_name]) if isinstance(i, str)]
tehn = [n for n, _, _ in teh]
missing = [(n, gt[seqid_name][n], i) for n, i in enumerate(gt[cat_name]) if n not in tehn]
print('missing', missing)
# 1/0
discos = teh
print('discos', discos)
# print('discos', [type(i) for n, id_, i in discos if not isinstance(i, str)])
# 1/0
overfits = [(n, gt[seqid_name][n], i) for n, i in enumerate(gt_sin) if isinstance(i, str) and 'fit' in i]
# discos = overfits
# discos = [(n, gt[seqid_name][n], i[:20]) for n, id_, i in discos if 'check' in i or 'fit' in i]
# print('discos', discos)

# non_nans = [n for n, id_, yes in discos if not pd.isna(gt[cat_name][n])]
# non_nans = [n for n, _ in enumerate(gt_sin) if not pd.isna(gt[cat_name][n])]
non_nans = [n for n, _, _ in discos]
trivials = [n  for n in non_nans if 'v' in gt[cat_name][n] or 'T' in gt[cat_name][n]]
exists = [n  for n in non_nans if 'x' in gt[cat_name][n] or 'E' in gt[cat_name][n]]
hards = [n  for n in non_nans if 'h' in gt[cat_name][n] or 'H' in gt[cat_name][n]]
# print([[(n, gt['sequence ID'][n]) for n in i] for i in [trivials, exists, hards, ]])
print(len(trivials), len(exists), len(hards))
print(sorted([gt[seqid_name][n] for n in trivials+exists+hards]))
# 1/0

# \section{method} matrix of fibonacci form:
a = [int(i) for i in pd.read_csv('cores_test.csv')['A000045']]
print(a)
# start, end = 5, 10
start, end = 5, 9
o = [print(f'{a[n]:<2} = c0 . {n}^3 + c1 . {a[n-1]:<3} + c2 . {a[n-2]*a[n-3]} + c_3 . {a[n-2]} + c4 . {n*a[n-5]**2}') for n in range(start, end)]

p = [print(f'{a[n]:<2} & = c_0 \cdot {n}^3 + c_1 \cdot {a[n-1]:<3} + c_2 \cdot {a[n-2]*a[n-3]:<3} + c_3 \cdot {a[n-2]} + c_4 \cdot {n}\cdot{a[n-5]:<3}^2 \\\\ ') for n in list(range(start+1, end)) + [14]]
print(a[5:15], a[14-2], a[14-3], a[14-2]*a[14-3])

m = [print( f'{n:<2}^3 &  {a[n - 1]:<3} &  {a[n - 2] * a[n - 3]:<3} &  {a[n - 2]} &  {n}\cdot{a[n - 5]:<3}^2 \\\\ ')
     for n in list(range(start + 1, end)) + [14]]
# 6^3 & 5  & 6 & 1^2  &  \\
#  7^3 & 8  & 15 & 1^2   \\
#  8^3 & 13 & 40 & 2^2   \\
#  & \vdots  & &  \\
#  14^3 & 233 & 89\cdot144 & 34^2  \\

1/0

# result analysis 2955 equations
seqid = 'A000045'
# dflin = pd.read_csv('linear_database_full.csv', low_memory=False)
dflin = pd.read_csv('linear_database_full.csv')
# print(dflin)  # for seq len = 200
seq = ed_fail_list[1][6:(6+7)]
print(dflin['A322829'][0])
# 1/0
# print(seq)
# print('ed_fail', ed_fail)
print('before dlfin')
# coefs = dflin['A000045'][0]
# print(coefs, len(coefs.strip('(').strip(')').split(',')), type(coefs))
# ed_fail_list = ['A000045', 'A001045']
# ed_fail_list = non_id_list
# ed_fails_ords = [(seq, len(dflin[seq[6:(6+7)]][0].strip('(').strip(')').split(','))) for seq in ed_fail_list]
# non_id_ords = [(seq, len(dflin[seq[6:(6+7)]][0].strip('(').strip(')').split(','))) for seq in non_id_list]
# print('\n'*4)

# small_fails = [(seq, o) for seq, o in ed_fails_ords if o <= 19]
# big_non_ids = [(seq, o) for seq, o in non_id_ords if o > 19]
# # biggie = [(seq, o) for seq, o in seqs if o >= 20]
# print('biggie fail', small_fails)
# print('biggie fail len', len(small_fails))
# print('biggie non_id', big_non_ids)
# print('biggie non_ids len', len(big_non_ids))
# # print('seqs', seqs)
# print('len fails', len(ed_fails_ords))
# print('len non ids', len(non_id_ords))
# print('eof')
#

# first 165 non_ids: ['00009_A000032.txt', '00010_A000035.txt', '00014_A000045.txt', '00017_A000058.txt', '00019_A000079.txt', '00021_A000085.txt', '00029_A000124.txt', '00030_A000129.txt', '00032_A000142.txt', '00034_A000166.txt', '00038_A000204.txt', '00039_A000217.txt', '00041_A000225.txt', '00042_A000244.txt', '00046_A000290.txt', '00047_A000292.txt', '00048_A000302.txt', '00051_A000326.txt', '00052_A000330.txt', '00054_A000396.txt', '00056_A000578.txt', '00057_A000583.txt', '00061_A000612.txt', '00067_A000798.txt', '00075_A001045.txt', '00077_A001057.txt', '00081_A001147.txt', '00088_A001333.txt', '00095_A001519.txt', '00097_A001699.txt', '00100_A001906.txt', '00106_A002275.txt', '00108_A002378.txt', '00111_A002530.txt', '00112_A002531.txt', '00114_A002620.txt', '00123_A004526.txt', '00130_A005408.txt', '00131_A005588.txt', '00133_A005843.txt', '00136_A006882.txt', '00158_A055512.txt']
