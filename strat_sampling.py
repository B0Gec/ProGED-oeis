from stratified_15555 import cx_dict
# # {'1': 1695, '2': 5723, '3': 4235, '4': 4175, '5': 1795, '6': 1734, '7': 879, '8': 1161, '9': 596, '10': 539, '11': 342, '12': 497, '13': 130, '14': 174, '15': 253, '16': 273, '17': 100, '18': 114, '19': 81, '20': 265}
# strata_sizes = [('1', 1695), ('2', 5723), ('3', 4235), ('4', 4175), ('5', 1795), ('6', 1734), ('7', 879), ('8', 1161), ('9', 596), ('10', 539), ('11', 342), ('12', 497), ('13', 130), ('14', 174), ('15', 253), ('16', 273), ('17', 100), ('18', 114), ('19', 81), ('20', 265)]

import math
sample_size = 100
# strata = [1, 2, ..., 20] = orders  # orders, look above
orders = [int(order) for order in cx_dict]
print(orders)
# popul = 15555
popul = sum([i[0] for i in cx_dict.values()])  # N
print(popul)
strata_sizes = [(order, cx_dict[str(order)][0]) for order in orders]  # list of N_h
print(strata_sizes)
print([(ord, size) for ord, size in strata_sizes if size < 100])
# strata_size_i*100/15555:

import random
strata_sample_sizes_float = [cx_dict[str(order)][0]*100/popul for order in orders]
strata_sample_sizes = [round(i, 2) for i in strata_sample_sizes_float]  # n_h
print(strata_sample_sizes, sum(strata_sample_sizes))
# print([i-math.floor(i) for i in strata_sample_sizes])
strata_sample_sizes = [round(i) for i in strata_sample_sizes]  # n_h
print(strata_sample_sizes, sum(strata_sample_sizes))

# avoiding empty strata and not summing up to stample size (=100).
# taking 2 extra units from elsewhere to fill 3 empty strata (total untill now is 99).
# manually_strata_sample_sizes = [round(i) for i in strata_sample_sizes_float]  # n_h
a = sorted([(n+1, round(size,2)) for n, size in enumerate(strata_sample_sizes_float)], key=lambda x: x[1]-math.floor(x[1]))
print(a)
# take from those 2 with lowest scores: (7, 3.55), (8, 4.69).
# final strata is:
strata_sample_sizes[7-1] -= 1
strata_sample_sizes[8-1] -= 1
strata_sample_sizes[-4:-1] = [1, 1, 1]
print('\nmanually produced sample sizes:')
print(strata_sample_sizes, sum(strata_sample_sizes))
manually_strata_units = strata_sample_sizes
# 1/0

chosen100 = [random.choices(cx_dict[str(order+1)][1], k=units) for order, units in enumerate(manually_strata_units)]
print(chosen100)
chosen100 = sum(chosen100, [])
print(len(chosen100), chosen100)
# 100 ['A177769', 'A013896', 'A022971', 'A304610', 'A013715', 'A158060', 'A199753', 'A304169', 'A189743', 'A185950', 'A057084', 'A041220', 'A015255', 'A153772', 'A051940', 'A132395', 'A157264', 'A140675', 'A123968', 'A182228', 'A253457', 'A068203', 'A182193', 'A277980', 'A190969', 'A065705', 'A041180', 'A187560', 'A304374', 'A157370', 'A021364', 'A021044', 'A085689', 'A213839', 'A111989', 'A183119', 'A168559', 'A016767', 'A128587', 'A256649', 'A192848', 'A268484', 'A253045', 'A208901', 'A122946', 'A256646', 'A084263', 'A178706', 'A291394', 'A105367', 'A214067', 'A295686', 'A219056', 'A293411', 'A077242', 'A251896', 'A080412', 'A106511', 'A097339', 'A000288', 'A054884', 'A129004', 'A008384', 'A299250', 'A268896', 'A267054', 'A213580', 'A069362', 'A116847', 'A115714', 'A047418', 'A108683', 'A255368', 'A107248', 'A112299', 'A026067', 'A055245', 'A164454', 'A047595', 'A107417', 'A235367', 'A027625', 'A168527', 'A290892', 'A041056', 'A055799', 'A133038', 'A123867', 'A185688', 'A327728', 'A230240', 'A035852', 'A024139', 'A029033', 'A169364', 'A042448', 'A107422', 'A010806', 'A072968', 'A001112']

# manual check:
# 1/1
# order1:
['A177769', 'A013896', 'A022971', 'A304610', 'A013715', 'A158060', 'A199753']
# yes,
# , ['A304169', 'A189743', 'A185950', 'A057084', 'A041220', 'A015255', 'A153772', 'A051940', 'A132395', 'A157264', 'A140675', 'A123968', 'A182228', 'A253457', 'A068203', 'A182193', 'A277980', 'A190969', 'A065705', 'A041180', 'A187560', 'A304374', 'A157370'], ['A021364', 'A021044', 'A085689', 'A213839', 'A111989', 'A183119', 'A168559', 'A016767', 'A128587', 'A256649', 'A192848', 'A268484', 'A253045', 'A208901', 'A122946', 'A256646', 'A084263'], ['A178706', 'A291394', 'A105367', 'A214067', 'A295686', 'A219056', 'A293411', 'A077242', 'A251896', 'A080412', 'A106511', 'A097339', 'A000288', 'A054884', 'A129004', 'A008384', 'A299250'], ['A268896', 'A267054', 'A213580', 'A069362', 'A116847', 'A115714', 'A047418'], ['A108683', 'A255368', 'A107248', 'A112299', 'A026067', 'A055245', 'A164454'], ['A047595', 'A107417', 'A235367'], ['A027625', 'A168527', 'A290892', 'A041056'], ['A055799', 'A133038'], ['A123867', 'A185688'], ['A327728'], ['A230240', 'A035852'], ['A024139'], ['A029033'], ['A169364'], ['A042448'], ['A107422'], ['A010806'], ['A072968'], ['A001112']]




