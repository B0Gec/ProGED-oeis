import pandas as pd
fname = 'header_database_linear.csv'
csv = pd.read_csv(fname)

from blacklist import blacklist
from blacklist import no_truth, false_truth
from exact_ed import truth2coeffs

print('fail:', [i for i in csv.columns if i in no_truth and i in false_truth])


# bigger = [i for i in csv.columns if len(truth2coeffs(csv[i][0])) > 20 if i not in blacklist]
# print(2)

from all_ids import all_ids
csv_filename = 'linear_database_full.csv'
full = pd.read_csv(csv_filename, low_memory=False, nrows=0)

from exact_ed import check_truth

# correct = ['A000045', 'A000004', 'A000008']
# for i in correct[:10]:
#     print(check_truth(i, csv_filename)[0][0])
for i in false_truth[:4]:
    check = check_truth(i, csv_filename, oeis_friendly=15)[0]
    print(i, check[1][:20])
    print(i, check[2][:20])
    print(check[0])


oeis_friendly=25
false_oeis_friendly = [id_ for id_ in false_truth
                       if not check_truth(id_, csv_filename, oeis_friendly=oeis_friendly)[0][0]]
print('oeis_friendly:', oeis_friendly)
print(len(false_oeis_friendly), false_oeis_friendly[:10])
print(false_oeis_friendly)

from exact_ed import check_eq_man
# check_eq_man(sp.Matrix())



