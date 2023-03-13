# helping calculating reconstruction error manually
import numpy as np
import pandas as pd
import os
main_path = 'results/check_pf/'
system = 'myvdp'
method = 'dso'
data_len = 'allonger'
leng = 2000 if data_len == 'allonger' else 100
snr = 'inf'
init = 0
path = f'{main_path}{system}{os.sep}{method}{os.sep}{leng}{os.sep}'
file = f'{path}pf_prettytable_{data_len}_s0_e1_a2_len{leng}_snr{snr}_init{init}_{system}.csv'
#
# csv = pd.read_csv(file)
# f = open(file, 'rb')
# print(f.read())
# f.close()
# os.getcwd()
# os.path.isdir(path)
# os.path.isfile(file)
# os.listdir(path)
# print(file)
#
# def re(dx, dx_, dy, dy_):


#x: 1, x, xy, x^2, 1
#y: 1, xy, x^2, 1
true_par_x = [20.0, -1.0, -1.0, 0.5, 1.0] + [0.0]*50
true_par_y = [10.0, -1.0, 0.5, 1.0] + [0.0]*50
def REbacres(x, y):
    x += [0]*(len(true_par_x)-len(x))
    y += [0]*(len(true_par_y)-len(y))
    return np.sqrt(sum((np.array(x) - np.array(true_par_x))**2) + np.sum((np.array(y) - np.array(true_par_y))**2))
print(REbacres([20, -1, -1, 0.5, 1], [10, -1, 0.5, 1]))


#x: y
#y: yx^2, x, y
true_par_x = [1.0] + [0.0]*50
true_par_y = [-2.0, -1.0, 2.0] + [0.0]*50
def remyvdp(x, y):
    x += [0]*(len(true_par_x)-len(x))
    y += [0]*(len(true_par_y)-len(y))
    return np.sqrt(sum((np.array(x) - np.array(true_par_x))**2) + np.sum((np.array(y) - np.array(true_par_y))**2))
print(remyvdp([1], [-2, -1, 2.0]))
# 1
print(remyvdp([1], [-1.996, -1, 1.996]))
print(remyvdp([0.981, 0.002], [-1.996, -1, 1.996]))
print(remyvdp([0.981, 0.002], [-1.996, -1, 1.996]))
# 0.057 -0.471 1 + -0.261 x + 0.057 y + 0.083 x^2 + 0.132 y^2 + -0.215 x*y + 0.071 x^3 + 0.113 y^3 + 0.298 x^2*y + -0.213 y^2*x', '-0.301 1 + -0.883 x + 0.493 y + -0.006 x^2 + 0.113 y^2 + -0.271 x*y + 0.066 x^3 + 0.141 y^3 + -0.758 x^2*y + -0.399 y^2*x']]
#x: y
#y: yx^2, x, y

