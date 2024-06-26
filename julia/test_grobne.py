import sympy as sp
from itertools import product
from functools import reduce
# p = product([1, 2, 3], [5, 6])
# print([i for i in p])
# 1/0

# Define the variables
x, y, z = sp.symbols('x y z')

# Define the polynomials
# sympy  docs:
# F = [x + 5*y - 2, -3*x + 6*y - 15]
# F = [x + 5*y - 2, -3*x + 6*y - 15]

# circle:
# pt1 =     [x - 1, y - 0]
# pt2 =   [x - 0, y - 1]
# pt3 = [x + 1, y + 0]
# pt4 =   [x - 0, y + 1]
# pt5 =   [x - 1/2, y - sp.sqrt(3)/2]

# fibonaci
# pt1 =     [x - 0, y - 1, z-1]
# pt2 =     [x - 1, y - 1, z-2]
# pt3 =     [x - 1, y - 2, z-3]
# pt4 =     [x - 2, y - 3, z-5]
# pt5 =     [x - 3, y - 5, z-8]

# combi = list(map(lambda x: x[0]*x[1], product(pt1, pt2)))
# combi = list(map(lambda x: x[0]*x[1], product(combi, pt3)))
# combi = list(map(lambda x: x[0]*x[1], product(combi, pt4)))
# combi = list(map(lambda x: x[0]*x[1], product(combi, pt5)))

# line y= 2x:
pt1 =     [x - 1, y - 2]
pt2 =     [x - 2, y - 4]
pt3 =     [x - 4, y - 8]
pt4 =     [x - 8, y - 16]
pt5 =     [x - 16, y - 32]
# pt4 =     [x - 2, y - 3, z-5]
# pt5 =     [x - 3, y - 5, z-8]
pts = [pt1, pt2, pt3, pt4, pt5]
# pts = [pt1, pt2, pt3, pt4]

# line y= x + 2:
pts = [
     [x - 0, y - 3],
     [x - 3, y - 6],
     [x - 6, y - 9],
     [x - 9, y - 12],
    [x - 12, y - 15],
    ]

# line y= x + 2:
pts = [
    [x - 0, y - 0],
    [x - 1, y - 1],
    [x - 2, y - 4],
    [x - 3, y - 9],
    [x - 4, y - 16],
    [x - 5, y - 25],
    [x - 6, y - 36],
    [x - 7, y - 49],
    [x - 8, y - 64],
    [x - 9, y - 81],
    [x - 10, y - 100],
]

# reduced = reduce(lambda x, y: x+y, [1, 2, 3, 4, 5], 1)
# reduced = reduce(lambda x, y: x+y, [1], 1)
# combi = list(map(lambda x: x[0]*x[1], product(pt1, pt2)))
reduced = reduce(lambda x, y:  list(map(lambda pair: pair[0]*pair[1], product(x, y))), pts[1:], pts[0])
# print('combinations:')
# print(reduced)

combi = reduced
print([i for i in combi])
F = combi
print([type(i) for i in F])
# 1/0

# Calculate the Gröbner basis
G = sp.groebner(F, x, y)

# Print the result
print("Gröbner basis:")
for poly in G:
    print(poly)

# F = [x + 5*y - 2, -3*x + 6*y - 15]
# groebner(F, x, y)