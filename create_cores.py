"""Explanation of creation of cores.csv:
     1. Select sequences with keywords core and nice and without keyword more, i.e.:
         "keyword:core keyword:nice -keyword:more" in oeis.org
     2. Reuse bseqs database in saved_new_bfile10000
     3. use only 200 terms of each sequence as for linear ones where we fill nan-s for missing terms.
"""

import pandas as pd
import numpy as np

# # fname = 'header_database_linear.csv'
# # from blacklist import blacklist
# # csv = pd.read_csv(fname)
# #
# # from exact_ed import truth2coeffs
# # def order(id_, csv, sign=-1):
# #     if id_ in blacklist:
# #         return 10**6*sign
# #     else:
# #         truth = csv[id_][0]
# #         coeffs = truth2coeffs(truth)
# #         return len(coeffs)
# #
# # print(order(csv.columns[0], csv))
# #
# # from ed_fails import ed_fails
# # from non_manuals import non_manuals
# #
# # cut = (9, 14, 15, 22)
# # cut = tuple(i-9 for i in cut)
# # ed_fails = [file[cut[2]:cut[3]] for file in ed_fails]
# # non_manuals = [file[cut[2]:cut[3]] for file in non_manuals]
# #
# #
# # # orders = [(i, order(i, csv)) for i in csv]
# # # print(max(orders))
# #
# # orders = [(i, order(i, csv)) for i in ed_fails]
# # less20 = [(i, order(i, csv)) for i in ed_fails if order(i, csv, sign=1) <= 20]
# # less20 = [(i, order(i, csv)) for i in non_manuals if order(i, csv, sign=1) > 20]
# # # print(orders)
# # print(less20)
# #
# # # more100 = [i for i in csv if order(i, csv) >= 100]
# # # more50 = [i for i in csv if order(i, csv) >= 50]
# # # more20 = [i for i in csv if order(i, csv) >= 20 and not i in ed_fails]
# # # # print(more100[:10])
# # # # print(more100[:1000], len(more100))
# # # # print(len(more50), more50[:1000])
# # # print(len(more20), more20[:1000])




from saved_new_bfile10000 import bseqs
# # core_nice_nomore = list(bseqs.keys())
# # print(core_nice_nomore)
from core_nice_nomore import cores
print(len(cores))
seqs = {i: bseqs[i][1:101] for i in cores if len(bseqs[i][1:200]) >= 100}
seqs = {i: bseqs[i][1:101] for i in cores if len(bseqs[i][1:201]) >= 200}
seqs = {i: bseqs[i][1:101] for i in cores if len(bseqs[i][1:201]) >= 15}
seqs = {i: bseqs[i][1:101] for i in cores if len(bseqs[i][1:205]) >= 12}
# seqs = {i: bseqs[i][1:201] + [0 for i in range(200-len(bseqs[i][1:201]))] for i in cores}
seqs = {i: [int(j) for j in bseqs[i][1:201]] + [np.nan for i in range(200-len(bseqs[i][1:201]))] for i in cores}
print(bseqs['A000043'][1:40])
# 1/0


print(len(seqs))
# print(seqs['A000045'])
# print(seqs[0][1:200])
# print([(len(seqs[i])) for i in seqs])
# print([i for i in seqs if len(seqs[i]) != 200], 'counter')
# # print([(i, len(seqs[i])) for i in seqs if len(seqs[i]) <= 198])
# print([(i, len(bseqs[i])) for i in bseqs if len(bseqs[i]) <= 198])
# # # print([len(seqs[i]) for i in seqs if len(seqs[i]) <= 198])
# # print(seqs['A000045'])
df = pd.DataFrame(seqs)
df_sorted = df.sort_index(axis=1)
print(df_sorted.head())
print(df_sorted)
print(df_sorted['A000043'][:50])

# # csv_filename = "cores.csv"
# csv_filename = "cores_test.csv"
# csv_filename = "linear_database_full.csv"
# # df_sorted.to_csv(csv_filename, index=False)
# df = pd.read_csv(csv_filename)
# print(df)
# print('test')
# # print(df['A000043'][:50])
# # cons = [i for i in df if df[i].isnan().any()]
# print(type(df['A000004']))
# # print(cons)
#
#
#
#
