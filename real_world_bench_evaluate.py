"""
Evaluation of "Real-world" benchmark for exacte equation discovery.

For Diofantos paper:
5 datasets with 10 equations:
- bezut id
- Pell eg
add of det.
"""
import numpy as np


# 2.) Evaluation
################
import pandas as pd
import sympy as sp


# def Diofantos_csv
from exact_ed import diofantos, grid_sympy

csv = pd.read_csv('real-bench/wheel.csv')
# print(csv.head())
vars = list(csv.columns)
# y = vars[-1]
# rhs_obs_vars = vars[:-1]
# all_obs_vars = vars
# print('vars!!!!', y, rhs_obs_vars)
print('vars!!!!', vars)
print()
# 1/0

M = csv.to_numpy()
M = sp.Matrix(csv.to_numpy())
# import random
# print(random.shuffle(M))
# print(M)
# 1/0
scale = 400
M = M[:scale, :]
print(M[:5, :].__repr__())
print('start eq. 10')

# vector, eq = diofantos(M, 2, vars)
# print(eq)
print()
# Delta(W_n) = n
# print(csv)

# # eq 9
# print('\nstart eq. 9')
# print()
# target = -2
def target_prep(M, vars, target):
    vars = vars[:target] + vars[target + 1:] + [vars[target]]
    M = sp.Matrix.hstack(M[:, :target], M[:, target + 1:], M[:, target])
    return M, vars
#
# M_9, vars_9 = target_prep(M, vars, target)
# print(vars_9)
# print(M_9[:5, :].__repr__())
# M_9 = sp.Matrix.hstack(M_9[:, :1], M_9[:, -1])
# vars_9 = vars_9[:1] + [vars_9[-1]]
# print(vars_9)
# print(M_9[:5, :].__repr__())
#
# vector, eq = diofantos(M_9, 2, vars_9)
# print(eq)
# print()
# # delta(W_n) = -n + V(W_n) + 2
# #       -> delta(W_n) = -n + n+1 + 2 = 3
# # also using only first and delta(W_n) column we discovered delta(W_n) = 3

# eq 8
print('\nstart eq. 8\n')
target = -3
M_8, vars_8 = target_prep(M, vars, target)
print(vars_8)
print(M_8[:5, :].__repr__())
vector, eq = diofantos(M_8, 1, vars_8)
print(eq)
# Edges(W_n) = n + Delta(W_n)
#   from before Delta(W_n) = n
#      -> Edges(W_n) = n + n = 2n


