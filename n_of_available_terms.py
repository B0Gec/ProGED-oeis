import pandas as pd
import matplotlib.pyplot as plt

task_limit = 1000
# task_limit = 200

csv_filename = 'linear_database_newbl.csv'
# csv_filename = 'cores_test.csv'

# chosen:
# first 17 non_manuals:
non_mans = ['00011_A000064.txt', '00051_A000363.txt', '00062_A000478.txt', '00063_A000486.txt', '00064_A000487.txt', '00065_A000498.txt', '00068_A000539.txt', '00069_A000540.txt', '00070_A000541.txt', '00071_A000542.txt', '00072_A000543.txt', '00075_A000563.txt', '00076_A000565.txt', '00086_A000584.txt', '00087_A000596.txt', '00088_A000597.txt', '00092_A000771.txt']
non_mans =  ['00011_A000064.txt', '00051_A000363.txt', '00062_A000478.txt', '00063_A000486.txt', '00064_A000487.txt', '00065_A000498.txt', '00068_A000539.txt', '00069_A000540.txt', '00070_A000541.txt', '00071_A000542.txt', '00072_A000543.txt', '00075_A000563.txt', '00076_A000565.txt', '00086_A000584.txt', '00087_A000596.txt', '00088_A000597.txt', '00092_A000771.txt']
non_mans = ['00011_A000064.txt', '00065_A000498.txt', '00071_A000542.txt', '00088_A000597.txt', '00108_A000971.txt', '00109_A000972.txt', '00115_A001017.txt', '00167_A001301.txt', '00168_A001302.txt', '00171_A001305.txt', '00173_A001313.txt', '00185_A001401.txt', '00186_A001402.txt', '00195_A001496.txt']
non_mans = ['00011_A000064.txt', '00065_A000498.txt', '00071_A000542.txt', '00088_A000597.txt', '00108_A000971.txt', '00109_A000972.txt', '00115_A001017.txt', '00167_A001301.txt', '00168_A001302.txt', '00171_A001305.txt', '00173_A001313.txt', '00185_A001401.txt', '00186_A001402.txt', '00195_A001496.txt', '00313_A001977.txt', '00314_A001996.txt', '00316_A002015.txt', '00326_A002082.txt', '00374_A002526.txt', '00375_A002528.txt', '00391_A002622.txt', '00394_A002626.txt', '00403_A002727.txt', '00409_A002889.txt', '00417_A003082.txt', '00426_A003402.txt', '00427_A003404.txt', '00534_A004500.txt', '00656_A005783.txt', '00657_A005784.txt', '00718_A006148.txt', '00740_A006333.txt', '00741_A006334.txt', '00804_A006980.txt', '00806_A007010.txt', '00832_A007487.txt', '00864_A007786.txt', '00881_A007990.txt', '00883_A007994.txt', '00896_A008364.txt', '00897_A008377.txt', '00898_A008379.txt', '00899_A008382.txt', '00903_A008396.txt', '00908_A008454.txt', '00909_A008455.txt', '00910_A008456.txt', '00920_A008503.txt', '00921_A008504.txt', '00922_A008505.txt', '00923_A008506.txt', '00931_A008583.txt', '00932_A008584.txt', '00974_A008628.txt', '00975_A008629.txt', '00976_A008630.txt', '00977_A008631.txt', '00978_A008632.txt', '00979_A008633.txt', '00980_A008634.txt', '00981_A008635.txt', '00982_A008636.txt', '00983_A008637.txt', '00984_A008638.txt', '00985_A008639.txt', '00986_A008640.txt', '00987_A008641.txt', '00999_A008666.txt']
non_mans = ['00011_A000064.txt', '00065_A000498.txt', '00071_A000542.txt', '00088_A000597.txt', '00108_A000971.txt', '00109_A000972.txt', '00115_A001017.txt', '00167_A001301.txt', '00168_A001302.txt', '00171_A001305.txt', '00173_A001313.txt', '00185_A001401.txt', '00186_A001402.txt', '00195_A001496.txt', '00313_A001977.txt', '00314_A001996.txt', '00316_A002015.txt', '00326_A002082.txt', '00374_A002526.txt', '00375_A002528.txt', '00391_A002622.txt', '00394_A002626.txt', '00403_A002727.txt', '00409_A002889.txt', '00417_A003082.txt', '00426_A003402.txt', '00427_A003404.txt', '00534_A004500.txt', '00656_A005783.txt', '00657_A005784.txt', '00718_A006148.txt', '00740_A006333.txt', '00741_A006334.txt', '00804_A006980.txt', '00806_A007010.txt', '00832_A007487.txt', '00864_A007786.txt', '00881_A007990.txt', '00883_A007994.txt', '00896_A008364.txt', '00897_A008377.txt', '00898_A008379.txt', '00899_A008382.txt', '00903_A008396.txt', '00908_A008454.txt', '00909_A008455.txt', '00910_A008456.txt', '00920_A008503.txt', '00921_A008504.txt', '00922_A008505.txt', '00923_A008506.txt', '00931_A008583.txt', '00932_A008584.txt', '00974_A008628.txt', '00975_A008629.txt', '00976_A008630.txt', '00977_A008631.txt', '00978_A008632.txt', '00979_A008633.txt', '00980_A008634.txt', '00981_A008635.txt', '00982_A008636.txt', '00983_A008637.txt', '00984_A008638.txt', '00985_A008639.txt', '00986_A008640.txt', '00987_A008641.txt', '00999_A008666.txt', '01001_A008668.txt', '01007_A008674.txt', '01008_A008675.txt', '01011_A008678.txt', '01014_A008681.txt', '01100_A008862.txt', '01101_A008863.txt', '01103_A008881.txt', '01117_A009641.txt', '01118_A009694.txt', '01119_A009714.txt', '01151_A010034.txt', '01217_A010801.txt', '01218_A010802.txt', '01219_A010803.txt', '01220_A010804.txt', '01221_A010805.txt', '01222_A010806.txt', '01223_A010807.txt', '01224_A010808.txt', '01225_A010809.txt', '01226_A010810.txt', '01227_A010811.txt', '01228_A010812.txt', '01229_A010813.txt', '01329_A011820.txt', '01330_A011821.txt', '01333_A011850.txt', '01334_A011851.txt', '01335_A011853.txt', '01346_A011869.txt', '01351_A011874.txt', '01353_A011876.txt', '01354_A011877.txt', '01355_A011878.txt', '01357_A011880.txt', '01358_A011881.txt', '01359_A011882.txt', '01361_A011884.txt', '01373_A011901.txt', '01376_A011907.txt', '01390_A011927.txt', '01392_A011929.txt', '01396_A011933.txt', '01398_A011937.txt', '01400_A011939.txt', '01402_A011942.txt', '01691_A014095.txt', '01784_A014670.txt', '01790_A014718.txt', '01806_A014796.txt']


ids = [i[6:13] for i in non_mans]
print(len(ids), ids)

df = pd.read_csv(csv_filename, low_memory=False, usecols=ids)
for id_ in ids:
    print(id_, len(df[id_][0]))
# print(df["A000129"].dropna())
# print(df["A000129"].dropna()[1:20])

big = 100
big = 30
orderbigs = [i for i in ids if len(df[i][0]) < big]
print(f'\norders less than {big}: {len(orderbigs)}', orderbigs)

1/0

cols = pd.read_csv(csv_filename, low_memory=False, nrows=0)
cols = list(cols)[:task_limit]
print(cols)
# 1/0

df = pd.read_csv(csv_filename, low_memory=False, usecols=cols)
ids = [id_ for id_ in df]
print(df[ids[0]].dropna())
# print(df['A000043'])
# print(df['A000043'].dropna())
print(len(df[ids[0]].dropna()))
avails = [(df[i], len(df[i].dropna())) for i in df]
avails = [(i, len(df[i].dropna())) for i in df]
print(avails[:10])
# print(list(df['A190528']))
thresh = 30
# thresh = 120

less_terms = [(id_, avail) for id_, avail in avails if avail <= thresh]
print(less_terms[:10])
scarcest = sorted(less_terms, key=lambda x: (x[1], x[0]))  # ascending
print(scarcest[:10])
print(scarcest)
# [('A190528', 3), ('A204419', 6), ('A135982', 7), ('A145205', 8), ('A145206', 8), ('A145207', 8), ('A145309', 8), ('A182990', 8), ('A202280', 8), ('A324271', 8), ('A351237', 8), ('A138826', 9), ('A145306', 9), ('A145307', 9), ('A227040', 9), ('A227274', 9), ('A227275', 9), ('A031982', 10), ('A144863', 10), ('A145320', 10), ('A145333', 10), ('A178297', 10), ('A201226', 10), ('A203627', 10), ('A220983', 10), ('A220984', 10), ('A227110', 10), ('A227137', 10), ('A227138', 10), ('A351239', 10), ('A017412', 11), ('A017423', 11), ('A017424', 11), ('A017435', 11), ('A017436', 11), ('A017544', 11), ('A017555', 11), ('A017556', 11),
print([i for i,j in scarcest])
# for i,j in scarcest[:50]:
#     print(f'window.open(\'https://oeis.org/{i}/b{i[1:]}.txt\');')
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

gts = [(i, len(df[i][0].split(',')), df[i][0]) for i in df]
print(gts)
order = 2
easies = []
for order in range(1, 10):
    order_n = [(task, length) for task, length, _ in gts if length == order]
    easies.append(order_n)
print(easies)
1/0

lens = [(l, sum([1 for _, length, _ in gts if length == l ])) for l in list(set([length for _, length, _ in gts]))]
print(lens)
print(len(lens), sum([j for i,j in lens]))
len_thresh = 21
len_thresh = 7
len_thresh = 4
bigger = [(l, popul) for l, popul in lens if l <= len_thresh]
print(bigger)
print(len(bigger), sum([j for i,j in bigger]))

showlim = 10
