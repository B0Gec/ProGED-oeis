import pandas as pd

# csv = pd.read_csv('ground_truth-backup-results-oct24.csv')
# print(csv)
# print(csv.columns[6])
# print(list(csv.columns)[:6])
# print(csv['MB (Moeller-Buchberger)'])
#
import numpy as np
# gt = pd.read_csv('gt1125.csv')
# gt = pd.read_csv('ground_truth - ground_truth918_3.csv')
# gt = pd.read_csv('ground_truth - ground_truth922.csv')
# gt = pd.read_csv('gt919.csv')
gt = pd.read_csv('ground_truth-backup-results-oct24.csv')

gt_sin = gt['SINDy']
gt_mb = gt['MB (Moeller-Buchberger)']
# gt_sin = gt['Diofantos [disco., outputed]']

print(gt)
print(gt.columns)
# print(gt_sin[112], gt[gt.columns[0]][112])
# 1/0

# print(gt_sin[1], type(gt_sin[1]), )
# print(gt_sin == np.nan)
cat_name = 'cathegory (trivial [T]/exists [E]/hard [H])'  # x, h, v
print('cat', gt[cat_name][1], type(gt[cat_name][1]))
seqid_name = 'Unnamed: 0'
discos = [(n, gt[seqid_name][n], i) for n, i in enumerate(gt_sin) if isinstance(i, str) and 'yes' in i]
teh = [(n, gt[cat_name][n], i) for n, i in enumerate(gt[cat_name]) if isinstance(i, str)]
tehn = [n for n, _, _ in teh]
missing = [(n, gt[seqid_name][n], i) for n, i in enumerate(gt[cat_name]) if n not in tehn]
print('missing', missing)
# 1/0
discos = teh
print('discos', discos)
# print('discos', [type(i) for n, id_, i in discos if not isinstance(i, str)])
# 1/0
overfits = [(n, gt[seqid_name][n], i) for n, i in enumerate(gt_sin) if isinstance(i, str) and 'fit' in i]
# discos = overfits
# discos = [(n, gt[seqid_name][n], i[:20]) for n, id_, i in discos if 'check' in i or 'fit' in i]
# print('discos', discos)

# non_nans = [n for n, id_, yes in discos if not pd.isna(gt[cat_name][n])]
# non_nans = [n for n, _ in enumerate(gt_sin) if not pd.isna(gt[cat_name][n])]
non_nans = [n for n, _, _ in discos]
trivials = [n  for n in non_nans if 'v' in gt[cat_name][n] or 'T' in gt[cat_name][n]]
exists = [n  for n in non_nans if 'x' in gt[cat_name][n] or 'E' in gt[cat_name][n]]
hards = [n  for n in non_nans if 'h' in gt[cat_name][n] or 'H' in gt[cat_name][n]]
# print([[(n, gt['sequence ID'][n]) for n in i] for i in [trivials, exists, hards, ]])
print(len(trivials), len(exists), len(hards))
print(sorted([gt[seqid_name][n] for n in trivials+exists+hards]))
# 1/0
