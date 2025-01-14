"""
Count results for cores.
"""
import pandas as pd
import numpy as np

csv = pd.read_csv('ground_truth-backup-results-oct24.csv')
# csv = pd.read_csv('ground_truth-backup-results-16oct24.csv')
# csv = pd.read_csv('../Downloads/ground_truth-backup-results-16oct24-2.csv')
csv = pd.read_csv('ground_truth-backup-results-16oct24-3.csv')

method = 'MB (Moeller-Buchberger)'
mb = csv[method]
# alls = len(mb)

print('all rows:', len(mb))
print(csv.columns[0])
seqsonly =  len([i for i in csv[csv.columns[0]] if not pd.isna(i)])
print( 'all sequence rows:', seqsonly)
mb = mb[0:seqsonly]
print(len(mb))
# all_ids = [(csv.iloc[n, 0], i) for n, i in enumerate(mb)]

ss_l = [i for i in mb if i == 's']
# syesscoma = [i for i in mb if (i == 's' or ('yes' in str(i)) or str(i)[:2] == 's,')]
# count_l = [(csv.iloc[n, 0], i) for n, i in enumerate(mb) if (i == 's' or ('yes' in str(i)) or str(i)[:2] == 's,' or 'a(n) = n*(n+1)*(2*n+1)/6' in str(i)[:25] or str(i) == 'n')]  # 'n' = a2426  'n(n+1)(2n+1)/6' = a000330
count_l = [(csv.iloc[n, 0], i) for n, i in enumerate(mb) if (i == 's' or ('yes' in str(i)) or str(i)[:2] == 's,' )]
print(count_l)
ss = len(ss_l)
count = len(count_l)

nany = mb[0]
nans_l = [ i for i in mb if pd.isna(i) ]
nos_l = [ i for i in mb if i == 'no' ]
no_in = [ i for i in mb if 'no' in str(i).lower() ]
maybe = [ i for i in mb if 'maybe' in str(i) ]
# older, non-updated file:
# specify = [ i for i in mb if 'a(n) = -1*(a(n-2)^3 -2*' in str(i)[:25] ]  #A002658
# specify += [ i for i in mb if 'a(n) * a(n-1) + a(n) * a(n-2) ' in str(i)[:30] ]  # A000002, new useless formulas.
# for i in specify:
#     print('spec', i)

print('specials:')  # only informative:
for j in [(n, csv.iloc[n, 0], i) for n, i in enumerate(mb) if ('a(n) = n*(n+1)*(2*n+1)/6' in str(i)[:25] or  # A000330
                                                              str(i) == 'n' or  # 'n' = a2426
                                                              'a(n) = -1*(a(n-2)^3 -2*' in str(i)[:25] or  #A002658
                                                              'a(n) * a(n-1) + a(n) * a(n-2) ' in str(i)[:30])]:  #A000002
    print(j)

print()
print(1, mb[1])
print(52, mb[52])
print(109, mb[109])
print(116, mb[116])


# 1/0
nans = len(nans_l)
nos = len(nos_l)
no_ins = len(no_in)
maybes = len(maybe)
# specific = len(specify)
specific = 0
print(count)
# print('ss:', ss)

print()
print('captured:', count)
print('nans:', nans, 'nos:', nos, '(no in)s:', no_ins, 'maybes:', maybes)
print('non nans and nos:', len(mb) - nans - nos)
print('non nans and no ins:', len(mb) - nans - no_ins)
print('reduced', len(mb) - nans - no_ins - maybes - specific)
if count == (len(mb) - nans - no_ins - maybes - specific):
    print(f'\n  Counts match, so discovered equations for {method} seems to be: {count}\n')

print(f'results from mbcore0 show 45 successes out of {len(mb)} core sequences for MB method with '
      f'10 n_more_terms and an input having n_of_terms = 2*order + 10.')


# len(mb)
# print(mb)
trivials, exists, hards = tuple([(csv.iloc[n, 0]) for n, _ in enumerate(mb) if
            any([word in str(csv['cathegory (trivial [T]/exists [E]/hard [H])'][n]) for word in cat]) ]
            for cat in [('v', 'T'), ('x', 'E'), ('h', 'H', 'nan')])
# discos = [n for n, i in enumerate(mb) if 'yes' in str(i)]
print(len(count_l), count_l)
discos = [[i for i, _ in count_l if i in cat] for cat in [trivials, exists, hards]]
print('by cat:', [len(i) for i in discos], discos)
print('hard:', discos[2])  # Catalan numbers, accidentally put into hard category, although in paper explicitely say catalan have simple equation.

print()
print('trivials:', trivials)
print('exists:', hards)
print('hards:', hards)
# 1/0
print(f'  trivials:', len(trivials), f'\n  exists:', len(exists), f'\n  hards:', len(hards))
print('sum check', len(trivials) + len(exists) + len(hards),  len(mb))
# print([csv['cathegory (trivial [T]/exists [E]/hard [H])'][n] for n, i in enumerate(mb) if csv['cathegory (trivial [T]/exists [E]/hard [H])'][n] not in trivials + exists + hards])
# print([y for n, i in enumerate(mb) if (y := (csv.iloc[n, 0], i)) not in trivials + exists + hards])


1/0




from results_mb import seq_task_ids

print('targeted_taskids:', len(seq_task_ids))

ids = [i[0] for i in seq_task_ids]
count_ids = [i for i, _ in count_l]
# idss = [i for i in seq_task_ids]
# print(idss)
# 1/0

print()
# print(ids)
untargeted = [i for i in count_ids if i not in ids]
fails_targeted = [i for i in ids if i not in count_ids]
print('untargeted:', untargeted)
print('targeted:', fails_targeted)
print('len targeted:', len(fails_targeted))
# for i in targeted_taskids:
#     print(i)


"""
fails:
(1, 'A000002', 'no, (a(n+2)*a(n+1)*a(n)/2 = a(n+2) + a(n+1) + a(n) - 3 )')
(35, 'A000169', 'a(n) = n^(n-1)')
(43, 'A000262', 'a(n) = (2*n-1)*a(n-1) - (n-1)*(n-2)*a(n-2)')
(50, 'A000312', 'a(n) = n^n')
(99, 'A001764', '2*n*(2n+1)*a(n) - 3*(3n-1)*(3n-2)*a(n-1) = 0')
(116, 'A002658', 'a(n + 1) = a(n) * (a(n) / a(n-1) + (a(n) + a(n-1)) / 2)')
(132, 'A005811', 'a(2n+1) = 2a(n) - a(2n) + 1, \na(4n) = a(2n), a(4n+2) = 1 + a(2n+1)')

Out ot this 7 eqs., 4 are of a kind we would expect for our algorithm to theoretically work.
all 4 are of degree 3.

One or two discovered equations were of degree 3.

Diofant or Sindy better than MB (the only one):
    - A000262
"""


for j in [(n, i, csv['ground truth'][n]) for n, i in enumerate(csv[csv.columns[0]]) if i in fails_targeted]:
    print(j)


