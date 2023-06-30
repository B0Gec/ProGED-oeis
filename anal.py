##
from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
fname = 'header_database_linear.csv'
from blacklist import blacklist
csv = pd.read_csv(fname)

from exact_ed import truth2coeffs
def order(id_, csv, sign=-1):
    if id_ in blacklist:
        return 10**6*sign
    else:
        truth = csv[id_][0]
        coeffs = truth2coeffs(truth)
        return len(coeffs)

print(order(csv.columns[0], csv))

# 2.) analyze how many of order below 20.
print(order(csv.columns[1], csv))
print(len([i for i in csv.columns if order(i, csv) <= 20]))
##

bigger = [i for i in csv.columns if order(i, csv) > 20]
print(len(bigger), 'len(bigger)')
bigger = bigger[:4000]
orders = [order(i, csv) for i in bigger]
print(max(orders))

##

start = dict()
def summary(till_now, i):
    ordr = order(i, csv)
    if not ordr in till_now:
        till_now[ordr] = 1
    else:
        till_now[ordr] = till_now[ordr] + 1
    return till_now

# print(reduce(summary, csv.columns, start))
bigger_dic = reduce(summary, bigger, start)
##

print(bigger_dic)
xs = sorted([i for i in bigger_dic])
ys = [bigger_dic[i] for i in bigger_dic]
print(max(xs))
plt.plot(xs, ys)
plt.show()
##

1/0

from ed_fails import ed_fails
from non_manuals import non_manuals

cut = (9, 14, 15, 22)
cut = tuple(i-9 for i in cut)
ed_fails = [file[cut[2]:cut[3]] for file in ed_fails]
non_manuals = [file[cut[2]:cut[3]] for file in non_manuals]


# orders = [(i, order(i, csv)) for i in csv]
# print(max(orders))

orders = [(i, order(i, csv)) for i in ed_fails]
less20 = [(i, order(i, csv)) for i in ed_fails if order(i, csv, sign=1) <= 20]
less20 = [(i, order(i, csv)) for i in non_manuals if order(i, csv, sign=1) > 20]
# print(orders)
print(less20)

# more100 = [i for i in csv if order(i, csv) >= 100]
# more50 = [i for i in csv if order(i, csv) >= 50]
# more20 = [i for i in csv if order(i, csv) >= 20 and not i in ed_fails]
# # print(more100[:10])
# # print(more100[:1000], len(more100))
# # print(len(more50), more50[:1000])
# print(len(more20), more20[:1000])


# from saved_new_bfile10000 import bseqs
# # core_nice_nomore = list(bseqs.keys())
# # print(core_nice_nomore)
# from core_nice_nomore import cores
# print(len(cores))
# seqs = {i: bseqs[i][1:101] for i in cores if len(bseqs[i][1:200]) >= 100}
# # print(len(seqs))
# # # print(seqs[0][1:200])
# # # print([(len(seqs[i])) for i in seqs])
# # # print([(i, len(seqs[i])) for i in seqs if len(seqs[i]) <= 198])
# # # print([len(seqs[i]) for i in seqs if len(seqs[i]) <= 198])
# # print(seqs['A000045'])
# df = pd.DataFrame(seqs)
# df_sorted = df.sort_index(axis=1)
# # print(df_sorted.head())
csv_filename = "cores.csv"
# df_sorted.to_csv(csv_filename, index=False)
df = pd.read_csv(csv_filename)
print(df)






