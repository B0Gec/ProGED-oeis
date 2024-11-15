"""
"Real-world" benchmark for exacte equation discovery.

For Diofantos paper:
5 datasets with 10 equations:
- bezut id
- Pell eg
add of det.
"""
import numpy as np


# 1.) Creation
############

# ds5: Wheel W_n
#
# 1 & 1 & Bézout's identity & $ax + by = c$ & $a, b, c, x, y$    \\
# \midrule
# 2 & 2 & Pell & $ax^2 - by^2 = c$ & $a, b, c, x, y$    \\
# \midrule
# % 3 & line & $x-a1/s1 = y-a2/s2 = z-a3/s3$ &  $x, y, z, a1, a2, a3, s2, s3$ \\
# 3 & \multirow{2}{*}{3} & add. of det. & $det(A)\cdot\det(B) = det(AB)$  &
# % \multirow{2}{*}{
# $\det(A), \det(B), $ \\
# 4 &  & homo. of det. & $\det(\alpha\cdot  A) = \alpha^n \cdot \det(A)$ & $\det(AB), \alpha, \det(\alpha\cdot A)$ \\
# \midrule
# 5 & \multirow{3}{*}{4} & add. of trace & $\tr(A + B) = \tr(A) + \tr(B)$ & \multirow{3}{*}{$\begin{array}{c}
#   \tr(A), \tr(B), \tr(A+B), \\
#   \tr(cA), \tr(AB), \tr(BA), \\
#   \lambda_1, \lambda_2, \lambda_3
# \end{array}$}    \\
# % 8 & & $tr(cA) = c*tr(A)$ & & & & & & \\
# 6 &  & com. of tr. & $\tr(AB) = \tr(BA)$   \\
# 7 & & trace as eigen. & $\tr(A) = \lambda_1+\lambda_2+\lambda_3$  \\
# \midrule
# % 7 & \multirow{3}{*}{5} &  bipart. ver. & $|V(K_{m,n})| = m+n$       & & & & & & \\
# % 8 & & bipart. edg. & $|E(K_{m,n})| = m*n$       & & & & & & \\
# % 9 & & bipart. col. & $\chi'(K_{m,n}) = \Delta(K_{m,n})$ & & & & & & \\
# % \midrule
# 8 & \multirow{3}{*}{5} &  edg. of wheel & $|E(W_n)| = 2n$   &
# \multirow{3}{*}{ $\begin{array}{c} n, |V(W_n)|, |E(W_n)|, \\
#                 \delta(W_n), \Delta(W_n)
#                 \end{array}$ }  \\
# 9 & & wheel's min. deg.  & $\delta(W_n) = 3$ \\
# 10 & & wheel's max. deg. & $\Delta(W_n) = n$  \\


def wheel(n):
    return n, n+1, 2*n, 3, n


dir_path = 'real-bench/'

def create_wheel():
    # wh_output = 'n, |V(W_n)|, |E(W_n)|, \delta(W_n), \Delta(W_n)\n'  # errors in sympy (sp.Matrix(E(W_n)))
    wh_output = 'n, V(W_n), Edges(W_n), delta(W_n), Delta(W_n)\n'
    for i in range(3, 100+3):
        wh_output += str(wheel(i))[1:-1] + '\n'

    print(wh_output)
    WRITE = False
    # WRITE = True
    if WRITE:
        with open(dir_path+'real_world_bench_ds5.csv', 'w') as f:
            f.write(wh_output + '\n')
    return

# create_wheel()
# 1/0

import re

def create_pitagora():
    with open(dir_path + 'pitagora-triplets.csv', 'r') as f:
        content = f.read()
        # pairs = re.findall(r'(\d{1,3}) ,(\d{1,3}) ,(\d{1,3}).*\n', content)
        pairs = re.findall(r'(\d{1,3}), (\d{1,3}), (\d{1,3}).*\n', content)

    # print(pairs[:13])
    pitagora_out = 'a, b, c, c^2\n'
    for triplet in pairs:
        # print(triplet)
        a, b, c = triplet
        # print(a, b, c)
        # print(int(a)**2 + int(b)**2, int(c)**2)
        pitagora_out += f'{a}, {b}, {c}, {int(c)**2}\n'
    print(pitagora_out)

    WRITE = False
    # WRITE = True
    if WRITE:
        with open(dir_path+'real_world_bench_ds1.csv', 'w') as f:
            f.write(pitagora_out + '\n')
    return

create_pitagora()
