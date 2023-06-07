comment = """
    This file creates no_truth, false_truth lists where: 
    no_truth = sequences with dots ... or some other hard to parse symbols in coefficients description
    false_truth = sequences whose truth written is incorrect.
"""

import pandas as pd

# csv_filename = 'linear_database_full.csv'
header_csv = 'header_database_linear.csv'
# csv = pd.read_csv(csv_filename, low_memory=False, nrows=1)
csv = pd.read_csv(header_csv, low_memory=False, nrows=1)
# print(csv.columns[:10])
# # csv.to_csv(header_csv, index=False)
#

output_string = f'all_ids = {str(list(csv.columns))}'
# print(output_string)

# # writo new files:
# out_fname = 'all_ids.py'
# f = open(out_fname, 'w')
# f.write(output_string)
# f.close()

# # check all_seqs:
# from all_ids import all_ids
# print(all_ids[:10], '\n', len(all_ids))

# zero_order = [id_ for id_ in csv]
def header2coefs(truth: str):

    # coeffs = [0] + truth[1:-1].split(',')[:len(csv[seq_id])-2]
    coeffs = truth[1:-1].split(',')
    # if coeffs != ['']:
    #     x = list(int(i) for i in coeffs)
    return coeffs


def ok_coefs(seq_id: str, csv: pd.DataFrame):

    truth = csv[seq_id][0]
    # coeffs = [0] + truth[1:-1].split(',')[:len(csv[seq_id])-2]
    replaced = truth.replace('{', '').replace('}', '')
    # replaced = truth
    peeled = replaced[1:-2] if replaced[-2] == ',' else replaced[1:-1]
    coeffs = peeled.split(',')
    if coeffs == ['']:
        return False
    elif '...' in truth or '"' in truth or '.' in truth:
        print(truth)
        return False
    # if '"' in truth:
    #     return False
    # elif seq_id in ('A025858', 'A246175', 'A025924'):
    #     return -2
    # elif seq_id in ('A356621'):
    #     return -2
        # elif seq_id in ('A029252', 'A356621'):
        # return False
    else:
        try:
            # try:
            #     if len(peeled) >= 1:
            #         1 + int(peeled[-1])
            # except Exception:
            #     print(truth)
            #     print(peeled)
            #     1 + int(peeled[-1])
            x = list(int(i) for i in coeffs)
        except Exception as e:
            print(seq_id, truth, coeffs)
            x = list(int(i) for i in coeffs)
        return True


# from blacklist import blacklist
# newblacklisted = [i for i in csv if ok_coefs(i, csv) == -2 and i not in blacklist]
comment = """No truth sequences are those that do not actually have truth written on website or they have only truth with
    sympols such as truth = (0,0, ..., 0, 1), etc."""
no_truth = [i for i in csv if not ok_coefs(i, csv)]


print('no_truth:')
print(no_truth[:10], len(no_truth))
output_string = f'no_truth = {no_truth}'

# writo new files:
out_fname = 'blacklist.py'  # v 18.4.2023
f = open(out_fname, 'r')
before = f.read()
print(before)
output_string = f'{before}\n{output_string}'
print(output_string)

# f.close()
# f = open(out_fname, 'w')
# f.write(output_string)
# f.close()

# test import:
from blacklist import no_truth
print(no_truth)
print(type(no_truth), len(no_truth))

# print(len(blacklist))
# print('new blacklisted:')
# print(len(newblacklisted), newblacklisted)

# output_string = f'blacklist = {blacklist + newblacklisted}'
# print(output_string)

# # writo new files:
# out_fname = 'blacklist.py'  # v 18.4.2023
# f = open(out_fname, 'w')
# f.write(output_string)
# f.close()


# # bug in ground truth string:
# empty = [i for i in csv if header2coefs(csv[i][0]) == ['']]
# useless = [i for i in csv if not ok_coefs(i, csv)]
# single = [i for i in csv if len(header2coefs(csv[i][0])) == 1]
# singlenonempty = [i for i in csv if i in single and not i in empty]
# zero_nontrivial = [i for i in csv if i in singlenonempty and i in empty]
#

from unsuccessful import unsuccessful_list
print('unsuccessful_list:')
print(unsuccessful_list[:10])

# unsucc_nonempty = [i for i in unsuccessful_list if not i in empty]
# print(unsucc_nonempty[:10])
# print(len(unsucc_nonempty))
#
# # print(out[:10])
# print(empty[:10])
# print(single[:10])
# print(singlenonempty[:10])
# print(zero_nontrivial)
# print(len(csv.columns), len(single), len(empty), len(empty) + len(singlenonempty))


# # check if old blacklist intersects empty groundtruth
# from blacklist import blacklist
#
# out = [i for i in csv if i in blacklist and i in empty]
# print(blacklist[:10])
# print('blacklist: ')
# print(out[:10])

# newblacklist = blacklist + empty
#
# output_string = f'blacklist = {newblacklist}'
#
# # writo new files:
# out_fname = 'blacklist.py'
# f = open(out_fname, 'w')
# f.write(output_string)
# f.close()
#



# # from functools import reduce
# #
# # summary = reduce(for_summary, files, (0, 0, 0, 0, 0, []))  # save all buggy ids
# #
# # def order_stat()
# #
# # more = [i for i in csv if len(header2coefs(csv[i][0])) == order ] for order in range()
# more = [i for i in csv if len(header2coefs(csv[i][0])) >= 20 ]
# print(len(more))
# more = [i for i in csv if len(header2coefs(csv[i][0])) >= 30 ]
# print(len(more))
# more = [i for i in csv if len(header2coefs(csv[i][0])) >= 20 ]
# print(len(more))



# Find new bugs:
print('\n New bugs here.')
from blacklist import blacklist
not_blacklisted = [i for i in unsuccessful_list if not i in blacklist]
print(not_blacklisted[:10])
