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

# check all_seqs:
from all_ids import all_ids
print(all_ids[:10], '\n', len(all_ids))

# zero_order = [id_ for id_ in csv]
def header2coefs(truth: str):
    coeffs = truth[1:-1].split(',')
    return coeffs

empty = [i for i in csv if header2coefs(csv[i][0]) == ['']]
single = [i for i in csv if len(header2coefs(csv[i][0])) == 1]
singlenonempty = [i for i in csv if i in single and not i in empty]
zero_nontrivial = [i for i in csv if i in singlenonempty and i in empty]
# print(out[:10])
print(empty[:10])
print(single[:10])
print(singlenonempty[:10])
print(zero_nontrivial)
print(len(csv.columns), len(single), len(empty), len(empty) + len(singlenonempty))


from blacklist import blacklist

out = [i for i in csv if i in blacklist and i in empty]
print(blacklist[:10])
print('blacklist: ')
print(out[:10])

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





