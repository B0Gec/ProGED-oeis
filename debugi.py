import sympy as sp
import numpy as np
import pandas as pd
import exact_ed as ed
#
# # ed.check_eq_man(sp.Matrix([[1], [-1], [1]]), 'A000085', pd.read_csv('cores_test.csv'), solution_ref=['n*a(n-2)', 'a(n-2)', 'a(n-1)'], header=False)
# ans = ed.check_eq_man(sp.Matrix([[14], [-49], [36]]), 'A074515', pd.read_csv('linear_database_clean.csv', usecols=['A074515']), solution_ref=['a(n-1)', 'a(n-2)', 'a(n-3)'], header=True)
# print(ans[0])
# print(ans[1])
# print(ans[2])
# 1/0

# ans = ed.check_eq_man(sp.Matrix([[13], [-36], [24]]), 'A074515', pd.read_csv('linear_database_clean.csv', usecols=['A074515']), solution_ref=['a(n-1)', 'a(n-2)', '1'], header=True)
# print(ans[1].__repr__())
# print(ans[2].__repr__())
# print([1+4**n+ 9**n for n in range(19)] == ans[1])
# print([ans[1][i] == ans[2][i] for i in range(len(ans[1]))])
# print(len(ans[1]), len(ans[2]))
# print(ans[0])

o_seq = pd.read_csv('linear_database_clean.csv', usecols=['A091881'])['A091881'].to_list()
o_seq = pd.read_csv('linear_database_clean.csv', usecols=['A091881'])['A091881']
print(o_seq[1:19])
# print(o_seq).iloc[3:20, :]
1/0
ans = ed.check_eq_man(sp.Matrix([[17], [-16]]), 'A091881', pd.read_csv('linear_database_clean.csv', usecols=['A091881']), solution_ref=['a(n-1)', 'a(n-2)'], header=True)
print(o_seq[1:])
print(ans[0])
print(ans[1])
print(ans[2])
1/0
seq = [float(i) for i in o_seq[1:]]
print(seq)
# 1/0

def an(an1, an2):
    return 17*an1 - 16*an2

rec = seq[:2]
for i in range(len(seq)-2):
    rec.append(an(rec[i+1], rec[i]))

print('seq vs. rec:')
print(seq)
print(rec)


# first 100 non_manuals: ['04132_A025938.txt', '09906_A074515.txt', '09908_A074517.txt', '11571_A091881.txt', '11572_A091883.txt', '11827_A094944.txt', '11939_A097068.txt', '13516_A114480.txt', '13922_A120465.txt', '13939_A120689.txt']

# print(pd.read_csv('linear_database_full.csv', usecols=['A074515'])['A074515'].tolist()[-1])
