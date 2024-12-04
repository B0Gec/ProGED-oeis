import pandas as pd
import matplotlib.pyplot as plt

csv_filename = 'linear_database_newbl.csv'
# csv_filename = 'cores_test.csv'
df = pd.read_csv(csv_filename, low_memory=False)
ids = [id_ for id_ in df]
print(df[ids[0]].dropna())
# print(df['A000043'])
# print(df['A000043'].dropna())
print(len(df[ids[0]].dropna()))
avails = [(df[i], len(df[i].dropna())) for i in df]
avails = [(i, len(df[i].dropna())) for i in df]
print(avails[:10])
print(list(df['A190528']))
thresh = 30
# thresh = 120

less_terms = [(id_, avail) for id_, avail in avails if avail <= thresh]
print(less_terms[:10])
scarcest = sorted(less_terms, key=lambda x: (x[1], x[0]))  # ascending
print(scarcest[:10])
print(scarcest)
# [('A190528', 3), ('A204419', 6), ('A135982', 7), ('A145205', 8), ('A145206', 8), ('A145207', 8), ('A145309', 8), ('A182990', 8), ('A202280', 8), ('A324271', 8), ('A351237', 8), ('A138826', 9), ('A145306', 9), ('A145307', 9), ('A227040', 9), ('A227274', 9), ('A227275', 9), ('A031982', 10), ('A144863', 10), ('A145320', 10), ('A145333', 10), ('A178297', 10), ('A201226', 10), ('A203627', 10), ('A220983', 10), ('A220984', 10), ('A227110', 10), ('A227137', 10), ('A227138', 10), ('A351239', 10), ('A017412', 11), ('A017423', 11), ('A017424', 11), ('A017435', 11), ('A017436', 11), ('A017544', 11), ('A017555', 11), ('A017556', 11),
print([i for i,j in scarcest])
for i,j in scarcest[:50]:
    print(f'window.open(\'https://oeis.org/{i}/b{i[1:]}.txt\');')
print(len(less_terms))
# cores: a58, a1699, a2658?, a6894
# 1/0


plot_dic = dict()
# plot_dic = {i: len(df[i].dropna()) for i in df}
for i in df:
    key = str(len(df[i].dropna()))
    plot_dic[key] = plot_dic.get(key, 0) + 1
    # print(i, len(df[i].dropna()))

# print(plot_dic)
print(len(plot_dic.keys()))

listed = [(key, value) for key, value in plot_dic.items()]
print('listed', listed[:10])
sorty = sorted(listed, key=lambda x: int(x[0]))
# sorty = sorty[:-1]
print(sorty[:10])
# sortie = [(key, value) for key, value in plot_dic.items() if int(key) <= 30]
# print(sortie)
# print([value for key, value in sortie])
# print(sum([value for key, value in sortie]))

x = [int(i[0]) for i in sorty]
y = [int(i[1]) for i in sorty]

# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.plot(plot_dic.keys(), plot_dic.values())
# plt.plot(x,y)
plt.bar(x,y)
# plt.bar(list(plot_dic.keys()), list(plot_dic.values()))
# plt.plot(list(plot_dic.keys()), list(plot_dic.values()))
plt.show()
