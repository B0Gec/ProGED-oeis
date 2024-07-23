"""
Testing naive method, by taking products for union of finite points in zero-dim variety,
although points lie on positive dimensional variety.
Idea: finite points -> polys -> products of polys -> grobner basis

Tested:
- fibonaci: NOT working! (not sure why)
- y = x + 3: working
- y = 2x : working?
- y = x + 3 in 3D (x, y, z) space: working
- y = x**2 : NOT working!
"""

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
pt1 =     [x - 0, y - 1, z-1]
pt2 =     [x - 1, y - 1, z-2]
pt3 =     [x - 1, y - 2, z-3]
pt4 =     [x - 2, y - 3, z-5]
pt5 =     [x - 3, y - 5, z-8]
fib = [pt1, pt2, pt3, pt4, pt5]
fib = [pt1, pt2, pt3]

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
pts2x = [pt1, pt2, pt3, pt4, pt5]
pts = pts2x
# pts = [pt1, pt2, pt3, pt4]

# line y= x + 3: (finds)
ptsxa3 = [
     [x - 0, y - 3],
     [x - 3, y - 6],
     [x - 6, y - 9],
     [x - 9, y - 12],
    [x - 12, y - 15],
    ]
pts = ptsxa3

# line y= x + 3: (finds)
ptsxa3iz = [
    [x - 0, y - 3, z - 0],
    [x - 3, y - 6, z - 0],
    [x - 6, y - 9, z - 0],
    [x - 9, y - 12, z - 0],
    [x - 12, y - 15, z - 0],
]

# pts21 = [
#     [x - 0, y - 1, z - 0],
#     [x - 1, y - 2, z - 0],
#     ]
pts = ptsxa3iz

# line y= x**2:
ptsx2 = [
    [x - 0, y - 0],
    [x - 1, y - 1],
    [x - 2, y - 4],
    [x - 3, y - 9],
    [x - 4, y - 16],
    [x - 5, y - 25],
    [x - 6, y - 36],
    [x - 7, y - 49],
    [x - 8, y - 64],
#     [x - 9, y - 81],
#     [x - 10, y - 100],
]
pts = ptsx2

# pts = fib
# pts = ptsxa3
# pts = pts2x

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
gens = [x, y, z][:len(pts[0])]
print(gens)
G = sp.groebner(F, x, y, z)

# checking out the output of groebner:
# rootes = sp.roots(5040*x + 5*y**4 - 154*y**3 + 1505*y**2 - 6396*y, y)
rootes = sp.real_roots(y**5 - 30*y**4 + 273*y**3 - 820*y**2 + 576*y)
print(rootes)
rootes = [ sp.real_roots(5040*x + 5*y**4 - 154*y**3 + 1505*y**2 - 6396*y) for y in [0, 1, 4, 9, 16]]
# 5040*x + 5*y**4 - 154*y**3 + 1505*y**2 - 6396*y
# we can see that it successfully eliminated variables in last equation,
# i.e. found the finite roots of the univariate polynomial - finite solutions.
print(rootes)
# 1/0

# Print the result
print("Gröbner basis:")
for poly in G:
    print(poly)

# F = [x + 5*y - 2, -3*x + 6*y - 15]
# groebner(F, x, y)
# test 2:
# counterex =
# G2 = sp.groebner(counterex, x, y)


1/0
# b) (trans)-plotting:

import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True

data1 = [
    [ 0,  3],
    [ 3,  6],
    [ 6,  9],
    [ 9, 12],
    [12, 15],
    ]

data2 = [
     [1, 2],
     [2, 4],
     [4, 8],
     [8, 16],
     [16, 32],
]

data3 = [
    [0, 0],
    [1, 1],
    [2, 4],
    [3, 9],
    [4, 16],
    ]
data = data3

# a = np.array([0, 1, 2, 3, 4])[[True, False, True, True, False]]
# # a = np.array([0, 1, 2, 3, 4])
# print(a)
# 1/0

fig, ax = plt.subplots(dpi=300)


plt.axis('equal')

# left, right, top, bottom = (-2, 10, 20, -10)
left, right, bottom, top  = (-15, 15, -4.5, 20.9)
osx = np.linspace(left, right, 100)
osy = np.linspace(bottom, top, 10000)
osyx = 0 * osy
osyy = osy
osxx = osx
osxy = 0 + 0* osx
# plt.plot(osyx, osyy, 'k--', linewidth=1)
# plt.plot(osxx, osxy, 'k--', linewidth=1)

ys = osx**2
sel = (ys < top)
ax.plot(osx[sel], ys[sel], 'r-', linewidth=0.6,
         label = '$y = x^2$ (ground truth)')
ax.legend()
# plt.show()


ax.plot([data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))] , 'o',
        label='Data points')
# ax.set_label('Data points')
# ax.legend()


ys = [0, 1, 4, 9, 16]
# osx = np.linspace(left, right, 100)
plts = [(osx, y+ 0*osx) for y in ys]
plt.plot(plts[0][0], plts[0][1], '-', color='orange',
         label= 'y^5 - 30y^4 + 273y^3 - 820y^2 + 576y = 0')
for pl in plts[1:]:
    plt.plot(pl[0], pl[1], '-', color='orange')


# # 0, 1, 4, 9, 16
# # This poly looks localy like horizontal lines:
# # ys = np.linspace(-2, 10, 100)
# # ys = np.linspace(left, right, 100)
# # ys = np.linspace(bottom, top, 100)
# ys = osy
# xs = ys**5 - 30*ys**4 + 273*ys**3 - 820*ys**2 + 576*ys
# sel = (xs > left) & (xs < right)
# ys = ys[sel]
# xs = xs[sel]
# plt.plot(xs, ys, 'c-', linewidth=0.61)
# plt.show()
# 1/0

# polyx = 5040*x + 5*y**4 - 154*y**3 + 1505*y**2 - 6396*y) for y in [0, 1, 4, 9, 16]]
# ys = np.linspace(-4, right, 100)
# ys = np.linspace(left, right, 100)
ys = osy
xs = (5*ys**4 - 154*ys**3 + 1505*ys**2 - 6396*ys)/(-5040)
# polyx = 5040*x + 5*y**4 - 154*y**3 + 1505*y**2 - 6396*y=0
plt.plot(xs, ys, 'm-',
         label= '    $5040x + 5y^4 - 154y^3 + 1505y^2 - 6396y = 0$')

# plt.show()
# 1/0
# ysx = nrange = np.linspace(left, right, 100)
# osyy = 0 * osyx
# plt.plot(osyx, osyy, 'k--')
# x = nrange = np.linspace(-10, 20, 100)
# y = 3 + x
# y = 2*x
# # 5040*x + 5*y**4 - 154*y**3 + 1505*y**2 - 6396*y
# # y**5 - 30*y**4 + 273*y**3 - 820*y**2 + 576*y = 0
# plt.plot(x, y, 'r-')
# plt.plot(osxx, -1+0*osxy, 'k--', linewidth=0.2)
# plt.plot(osyx, osyy, 'k--', linewidth=0.75)
plt.plot(osyx, osyy, 'k--', linewidth=0.6)
plt.plot(osxx, osxy, 'k--', linewidth=0.6)


# ax.legend(['', '', '$y = x^2$', 'Data points',
#            ' $5040x + 5y^4 - 154y^3 + 1505y^2 - 6396y = 0$ ',
#            ' $y^5 - 30y^4 + 273y^3 - 820y^2 + 576y = 0$ ',
#            ])
ax.legend(bbox_to_anchor=(0.343, 1.140187), loc='upper left')


plt.show()
