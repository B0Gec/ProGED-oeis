"""
Gathers results from running oeis on cluster.
results_oeis/good/01234567
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

import os
import re

# fname = 'results_oeis/good/01234567/34500_A000032.txt'
base_dir = "results_oeis/good/"
job_id = "01234567"
job_id = "36765084"
job_id = "36781342"  # 17.3. waiting 2 jobs
seq_file = '13000_A079034.txt'
job_dir = base_dir + job_id + '/'
fname = job_dir + seq_file
# print(os.listdir(exact_dir))



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

    if verbosity >= 1:
        print('n_of_seqs', re_all_stars)
        print('refound', re_found)
        print('reconst', re_reconst)
        print('reman', re_manual)

    n_of_seqs = int(re_all_stars[0])
    we_found = re_found == []
    if verbosity >= 1:
        print('we_found:', we_found)

    def truefalse(the_list):
        string = the_list[0]
        if string not in ('True', 'False'):
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

    # fs, ms, is, os = aggregated

    zipped = zip(aggregated, to_add)

    # return [ f+fs, m+ms, i+is, o+os for fs, ms, is, os in before]
    return tuple(map(lambda x: x[0] + x[1], zipped))



files = os.listdir(job_dir)
scale = 40
scale = 50100
files_debug = files[:scale]
files = files_debug
# print(files)

_a, _b, _, n_of_seqs = extract_file(job_dir + files[0])
print(n_of_seqs)

summary = reduce(for_summary, files, (0, 0, 0, 0, 0,))
corrected_sum = sum(summary[:4]) - sum(summary[4:])
print()
print(summary)
print(f'all files:{len(files)}, sum:{sum(summary)}, corrected sum: {corrected_sum}')
# print(f((1,2,3,4,), 'c'))

print()
print(f'Results: \n')


# all_seqs = 34000
sum(summary[:4]) - sum(summary[4:])
jobs_fail = n_of_seqs - len(files)  # or corrected_sum.

id_oeis, non_id, non_manual, fail, reconst_non_manual = summary
printout = f"""
    {id_oeis: >5} - (is oeis) - successfully found equations that are identical to the recursive equations written in OEIS (hereinafter - OEIS equation)
    {non_id: >5} - (non_id) - successfully found equations that are more complex than the OEIS equation 
    {non_manual: >5} - (non_manual) - successfully found equations that do not apply do not apply to test cases 
    {fail: >5} - (fail) - failure, no equation found. (but program finished smoothly, no runtime error)
    {reconst_non_manual: >5} - (reconst_non_manual) - fail in program, specifically reconstructed oeis and wrong on test cases.
    
    {jobs_fail: >5} - runtime errors - jobs failed
    {fail + jobs_fail: >5} - all fails  <--- (look this) ---
    {n_of_seqs: >5} - all sequences in our dataset
"""
print(printout)