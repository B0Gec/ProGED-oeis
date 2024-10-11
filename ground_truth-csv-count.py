import pandas as pd
import numpy as np

csv = pd.read_csv('ground_truth-backup-results-oct24.csv')

mb = csv['MB (Moeller-Buchberger)']
# alls = len(mb)

print('all rows:', len(mb))
print(csv.columns[0])
seqsonly =  len([i for i in csv[csv.columns[0]] if not pd.isna(i)])
print( 'all sequence rows:', seqsonly)
mb = mb[0:seqsonly]
print(len(mb))

ss_l = [i for i in mb if i == 's']
# syesscoma = [i for i in mb if (i == 's' or ('yes' in str(i)) or str(i)[:2] == 's,')]
count_l = [i for i in mb if (i == 's' or ('yes' in str(i)) or str(i)[:2] == 's,' or 'a(n) = n*(n+1)*(2*n+1)/6' in str(i)[:25] or str(i) == 'n')]  # 'n' = a2426
print(count_l)
ss = len(ss_l)
count = len(count_l)
nany = mb[0]
nans_l = [ i for i in mb if pd.isna(i) ]
nos_l = [ i for i in mb if i == 'no' ]
no_in = [ i for i in mb if 'no' in str(i).lower() ]
maybe = [ i for i in mb if 'maybe' in str(i) ]
specify = [ i for i in mb if 'a(n) = -1*(a(n-2)^3 -2*' in str(i)[:25] ]
specify += [ i for i in mb if 'a(n) * a(n-1) + a(n) * a(n-2) ' in str(i)[:30] ]
for i in specify:
    print('spec', i)

print(mb[111])
# print(mb[116])
# print(mb[1])

# 1/0
nans = len(nans_l)
nos = len(nos_l)
no_ins = len(no_in)
maybes = len(maybe)
specific = len(specify)
print(count)
# print('ss:', ss)
print('captured:', count)
print('nans:', nans, 'nos:', nos, '(no in)s:', no_ins, 'maybes:', maybes)
print('non nans and nos:', len(mb) - nans - nos)
print('non nans and no ins:', len(mb) - nans - no_ins)
print('reduced', len(mb) - nans - no_ins - maybes - specific)
