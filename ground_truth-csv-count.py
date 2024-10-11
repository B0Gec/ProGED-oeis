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
count_l = [i for i in mb if (i == 's' or ('yes' in str(i)) or str(i)[:2] == 's,')]
print(count_l)
ss = len(ss_l)
count = len(count_l)
nany = mb[0]
nans_l = [ i for i in mb if pd.isna(i) ]
nos_l = [ i for i in mb if i == 'no' ]
no_in = [ i for i in mb if 'no' in str(i) ]
maybe = [ i for i in mb if 'maybe' in str(i) ]
specify = [ i for i in mb if 'a(n) = -1*(a(n-2)^3 -2*' in str(i)[:25] ]
print(specify)

print(mb[116])

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
