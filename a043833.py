import math
from numpy import base_repr as br

def digit2num(d):
    d = 10 if d == 'A' else 11 if d == 'B' else 12 if d == 'C' else int(d)
    return d

upperl = 4
upperl = 2234567
a = [sum(map(digit2num, br(i, 13))) for i in range(172)]
# a = [sum(map(digit2num, br(i, 13))) for i in range(upperl)]
# a = [list(br(i, 13)) for i in range(10)]
# a = [str(i) for i in range(10)]
print(a)
for i in range(20):
    print(a[i*10:(i+1)*10])


# shalit
print(a[0:4], a[4])
# Sum_{n>=1} a(n)/(n*(n+1)) = 13 * log(13) / 12  (Shallit, 1984)
sh =  sum([a[n]/(n*(n+1)) for n in range(1, upperl)])
# print(list(map(lambda ul: sum([a[n]/(n*(n+1)) for n in range(1, ul)]), range(upperl))))
print(f'sh:{sh}, 13 * log(13) / 12 = {13 * math.log(13, math.e) / 12}')