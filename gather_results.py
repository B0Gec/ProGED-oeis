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
"""
##

import os
import re

from blacklist import no_truth, false_truth

# fname = 'results/good/01234567/34500_A000032.txt'
base_dir = "results/good/"
# job_id = "01234567"
# job_id = "36765084"  # old format
job_id = "36781342"  # 17.3. waiting 2 jobs  # to check
# job_id = "37100077"  # small
job_id = "37117747"
# job_id = "37117747_2"  # 23k succ
# job_id = "37256280"  # little of results
# job_id = "37256396"
# job_id = "37507488"
# job_id = "37507488_1"
# job_id = "37256396_6"
# job_id = "37680622"
# job_id = "37683622"
job_id = "32117747"
# job_id = "debug2"
job_id = "38095692"


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
def extract_file(fname, verbosity=VERBOSITY):

    if verbosity >= 1:
        print()
        print(f'extracting file {fname}')
    f = open(fname, 'r')
    content = f.read()
    # output_string = content
    if verbosity >= 2:
        print(content)

    re_all_stars = re.findall(r"scale:\d+/(\d+),", content)
    re_found = re.findall(r"NOT RECONSTRUCTED", content)
    re_reconst = re.findall(r"\n(\w{4,5}).+checked against website ground truth", content)
    re_manual = re.findall(r"\n(\w{4,5}).+\"manual\" check if equation is correct", content)
    # re_reconst = re.findall(r"\n(\w.+checked against website ground truth", content)

    # print(len(re_all_stars))
    # print(content)
    if len(re_all_stars) == 0:
        print('--pred')
        print(content)
        print('--po')
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

    # re_manual =
    # we_found, is_reconst, is_check, = not (re_found == []), bool(re_reconst[0]), bool(re_manual[0]),

    # for now:
    # we_found = is_check


    f.close()

    return we_found, is_reconst, is_check, n_of_seqs

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
    we_found, is_reconst, is_checked, _ = extract_file(job_dir + fname)

    id_oeis = is_reconst
    non_id = not is_reconst and is_checked
    non_manual = not is_checked and we_found
    fail = not we_found

    # makes no sense, e.g. is_reconst and not is_checked
    reconst_non_manual = is_reconst and not is_checked
    # non_manual = we_found and not is_checked

    buglist, job_bins, ed_fail_list, non_manual_list = aggregated[-4:]
    if debug:
        if reconst_non_manual:
            buglist += [fname]
            # print(aggregated, fname)
            # raise IndexError("Bug in code!")
        if black_check and non_manual:
            buglist += [fname]
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




    # summand = [f, m, i, o]
    to_add = (id_oeis, non_id, non_manual, fail, reconst_non_manual)


    # f, m, i, o = aggregated
    if VERBOSITY >= 1:
        print('to_add sum:', sum(to_add))
        print('to_add:', to_add)
        print('aggregated:', aggregated)
        if sum(to_add[:4]) - sum(to_add[4:]) > 1:
           print(' --- --- look here --- --- ')
           raise ArithmeticError
        # if sum(to_add[:4]) > 1:
        #     print(reconst_non_manual)
        #     print(' --- --- look here --- --- ')
        #     raise ArithmeticError




    zipped = zip(aggregated[:-2], to_add)
    counters = tuple(map(lambda x: x[0] + x[1], zipped))
    return counters + (buglist, job_bins, ed_fail_list, non_manual_list)

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
# # success_ids_pairs = list(map(lambda file: (file[cut[0]:cut[1]], file[cut[2]:cut[3]]), files))
# success_ids = list(map(lambda file: file[cut[2]:cut[3]], files))
from all_ids import all_ids
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
#
#
# # a. check for not blacklisted unsuccessful jobs
# from blacklist import blacklist, no_truth
# print(len(blacklist), len(set(blacklist)))
# # not_blacklisted = [(task, i) for task, i in unsuccessful if not i in blacklist]
# not_blacklisted = [i for i in all_ids if not i in blacklist and not i in success_ids]
# # not_blacklisted = [i for n, i in enumerate(all_ids) if not i in blacklist and not i in success_ids and i]
# print(not_blacklisted[:10], len(not_blacklisted))
# # not_missing_truth = [i for i in all_ids if not i in no_truth and not i in success_ids]
# # not_blacklisted_pairs = [(f"{task:0>5}", id_) for task, id_ in enumerate(all_ids)
# #                          if not id_ in blacklist and not id_ in success_ids]
# not_blacklisted_pairs = [(f"{all_ids.index(id_):0>5}", id_) for id_ in not_blacklisted]
# print(not_blacklisted_pairs[:10], len(not_blacklisted_pairs))
# # ['A044941', 'A053833', 'A055649'] 3
# # [('00184', 'A001299'), ('00185', 'A001300'), ('00186', 'A001301'), ('00187', 'A001302'), ('00195', 'A001313'), ('00196', 'A001314'), ('00198', 'A001319'), ('00222', 'A001492'), ('00347', 'A002015'), ('00769', 'A005813')] 1921
# # these are blacklisted due to false ground truth
# print('A055649' in blacklist)
# # 1/0
#
#
# # buglist = not_blacklisted
# # # output_string = f'successful_list = {successful_list}'
# # output_string = f'buglist = {buglist}'
# # # writo new files:
# # out_fname = 'buglist.py'
# # f = open(out_fname, 'w')
# # f.write(output_string)
# # f.close()
#



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

_a, _b, _, n_of_seqs = extract_file(job_dir + files[0])
# print(n_of_seqs)

# summary = reduce(for_summary, files, (0, 0, 0, 0, 0,))
summary = reduce(for_summary, files, (0, 0, 0, 0, 0, [], [0 for i in range(36)], [], []))  # save all buggy ids
# corrected_sum = sum(summary[:4]) - sum(summary[4:])
corrected_sum = sum(summary[:4]) - sum(summary[4:5])
print(corrected_sum)
print()
print(summary)
print(f'all files:{len(files)}, sum:{sum(summary[:4])}, corrected sum: {corrected_sum}')
# print(f((1,2,3,4,), 'c'))

print()
print(f'Results: ')


no_truth, false_truth = len(no_truth), len(false_truth)
ignored = no_truth + false_truth

# all_seqs = 34371
n_of_seqs_db = n_of_seqs
n_of_seqs = n_of_seqs - ignored
jobs_fail = n_of_seqs - len(files)  # or corrected_sum.

id_oeis, non_id, non_manual, ed_fail, reconst_non_manual, buglist, \
    job_bins, ed_fail_list, non_manual_list = summary
corrected_non_manual = non_manual - reconst_non_manual
all_fails = ed_fail + jobs_fail

official_success = id_oeis + non_id

printout = f"""
    {id_oeis: >5} = {id_oeis/n_of_seqs*100:0.3} % ... (id_oeis) ... successfully found equations that are identical to the recursive equations written in OEIS (hereinafter - OEIS equation)
    {non_id: >5} = {non_id/n_of_seqs*100:0.3} % ... (non_id) ... successfully found equations that are more complex than the OEIS equation 
    {non_manual: >5} = {non_manual/n_of_seqs*100:0.3} % ... (non_manual) ... successfully found equations that do not apply to test cases 
    {ed_fail: >5} = {ed_fail/n_of_seqs*100:0.3} % ... (fail) ... failure, no equation found. (but program finished smoothly, no runtime error)
    {reconst_non_manual: >5} = {reconst_non_manual/n_of_seqs*100:0.3} % ... (reconst_non_manual) ... fail in program, specifically: reconstructed oeis and wrong on test cases.
    ~~{corrected_non_manual:~>5} = {corrected_non_manual/n_of_seqs*100:0.3} % ... (corrected_non_manual = non_manual ... reconst_non_manual) ... non_manuals taking bug in my code into the account.~~

    {(id_oeis + non_id + corrected_non_manual + ed_fail): >5} ... (id_oeis + non_id + corrected_non_manual + fail) = sum
    
    
    {no_truth: >5} = {no_truth/n_of_seqs*100:0.3} % ... sequences with no ground truth -> missing jobs
    {false_truth: >5} = {false_truth/n_of_seqs*100:0.3} % ... sequences with false ground truth -> missing jobs
    {ignored: >5} = {ignored/n_of_seqs*100:0.3} % ... sequences with no or false ground truth -> all missing jobs by default
    {n_of_seqs_db: >5} ... all sequences in our dataset
    
    {jobs_fail: >5} = {jobs_fail/n_of_seqs*100:0.3} % ... runtime errors = jobs failed
    {all_fails: >5} = {all_fails/n_of_seqs*100:0.3} % ... all fails  <--- (look this) ---
    {n_of_seqs: >5} ... all considered sequences, i.e. in our dataset - ignored
    
    _______________________________________________________
    {id_oeis + non_id + corrected_non_manual + all_fails} ... (id_oeis + non_id + corrected_non_manual + all_fails) ... under the line check if it sums up
    {id_oeis + non_id + corrected_non_manual + ed_fail + ignored} ... (id_oeis + non_id + corrected_non_manual + ed_fails + ignored) ... under the line check if it sums up
    {id_oeis + non_id + corrected_non_manual + ed_fail + ignored + jobs_fail} ... (id_oeis + non_id + corrected_non_manual + ed_fails + ignored + jobs_fail) ... under the line check if it sums up
    
    {official_success: >5} = {official_success/n_of_seqs*100:0.3} % - official success (id_oeis + non_id)
    
    
    
    ==========================================================
    
    \\newcommand{{\\allSeqs}}{{{n_of_seqs}}}
    \\newcommand{{\\isOeis}}{{{id_oeis}}}
    \\newcommand{{\\nonId}}{{{non_id}}}
    \\newcommand{{\\nonMan}}{{{non_manual}}}
    \\newcommand{{\\edFail}}{{{ed_fail}}}
    \\newcommand{{\\jobFail}}{{{jobs_fail}}}
"""


print(printout)

n = 6
print(f'first {n} bugs:', buglist[:n])
print(len(buglist))
print(f'job bins (task_id= 0, 1, ... 34):', job_bins)
print(f'zipped job bins (task_id= 0, 1, ... 34):', [(n, i) for n, i in enumerate(job_bins)])
print(f'check bins: {len(files)} = {sum(job_bins)} ?')
for n, i in enumerate(job_bins):
    print(n, i)

print(f'first {n} ed_fails:', ed_fail_list[:n])
print(len(ed_fail_list))
print(f'first {n} non_manuals:', non_manual_list[:n])
print(len(non_manual_list))


def fname2id(fname):
    return re.findall("A\d{6}", fname)[0]

bugids = list(map(fname2id, buglist))
print(bugids)
print(buglist)
print('len(bugids)', len(bugids))
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

