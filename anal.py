import pandas as pd
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

from ed_fails import ed_fails
from non_manuals import non_manuals

cut = (9, 14, 15, 22)
cut = tuple(i-9 for i in cut)
ed_fails = [file[cut[2]:cut[3]] for file in ed_fails]


# orders = [(i, order(i, csv)) for i in csv]
# print(max(orders))

orders = [(i, order(i, csv)) for i in ed_fails]
less20 = [(i, order(i, csv)) for i in ed_fails if order(i, csv, sign=1) <= 20]
less20 = [(i, order(i, csv)) for i in non_manuals if order(i, csv, sign=1) <= 20]
# print(orders)
print(less20)

# more100 = [i for i in csv if order(i, csv) >= 100]
# more50 = [i for i in csv if order(i, csv) >= 50]
more20 = [i for i in csv if order(i, csv) >= 20 and not i in ed_fails]
# print(more100[:10])
# print(more100[:1000], len(more100))
# print(len(more50), more50[:1000])
print(len(more20), more20[:1000])


