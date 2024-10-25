"""
To be implemented, check results of josef urban and d'Ascoli.
"""

import re

import pandas as pd

datafile = 'julia/urb/solutions'

ids = ''
with open(datafile, 'r') as f:
    content = f.read()
    ids_ = re.findall(r'A\d{1,6}', content)

# print(content[:10])
print(len(ids_))
# print(ids)
print(ids_[:10])
idsix = [f'A{i[1:]:0>6}' for i in ids_]
print(idsix[:10])
# 1/0

csv_filename = 'linear_database_newbl.csv'
core_fname = 'cores_test.csv'
csv = pd.read_csv(csv_filename, low_memory=False, nrows=0)
cores = pd.read_csv(core_fname, low_memory=False, nrows=0)
all_ids = [i for i in csv.columns]
core_ids = [i for i in cores.columns]

from opponent_match import match
print(len(match))
print(len(all_ids))
# match =    [i for i in all_ids if i in idsix]
# not_match = [i for i in idsix if i not in all_ids]
# print(len(match))
# print(len(not_match))
# print(not_match[:20])
# print(match)
# print(not_match)


match_cores = [i for i in core_ids if i in idsix]
# not_match = [i for i in idsix if i not in all_ids]
print(len(match_cores))
print(match_cores)

mcores = [i for i in match_cores]
urb_cores = [re.sub(r'A0{1,6}', 'A', i) for i in mcores]
print(len(urb_cores))
print(urb_cores)
# print(101)
# 1/0
print('mcores'.replace(r'A0+', '334'))

for i in urb_cores:
    progs = re.findall(f'({i}).*\n(.+)', content)[0]
    # progs = re.findall('A000', content)
    print(progs)

new = [i for i in idsix if i not in core_ids]
# 14168 is the number!!!
print(14168 + 13819)



