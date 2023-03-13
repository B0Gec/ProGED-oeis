# helping calculating reconstruction error manually
import numpy as np
import pandas as pd
import os
import re

##
# true_x = [0.8, 1, 0.8, 1, -0.5, 2 * np.pi * 0.0015, 2]
# true_y = [0.6, 1, 4.53]
# x = [1, 1, 1, 1, 0, 0, 2.047]
# y = [0, 0, 4]
#
#
# def dM(true_x=true_x, true_y=true_y, x=x, y=y):
#     true_par_x = true_x + [0.0]*50
#     true_par_y = true_y + [0]*50
#     # true_par_x = [0.8, 1, 0.8, 1, -0.5, 2*np.pi*0.0015, 2] + [0.0]*50
#     # true_par_y = [0.6, 1, 4.53] + [0]*50
#     true_num_x = len(np.nonzero(true_par_x)[0])
#     true_num_y = len(np.nonzero(true_par_y)[0])
#
#     def REbacres(x, y):
#         x += [0]*(len(true_par_x)-len(x))
#         y += [0]*(len(true_par_y)-len(y))
#         return np.sqrt(sum((np.array(x) - np.array(true_par_x))**2) + np.sum((np.array(y) - np.array(true_par_y))**2))
#
#
#     num_model_correct_x = len(np.nonzero(x[:true_num_x])[0])
#     num_model_correct_y = len(np.nonzero(y[:true_num_y])[0])
#     num_model_wrong_x = len(np.nonzero(x[true_num_x:])[0])
#     num_model_wrong_y = len(np.nonzero(y[true_num_y:])[0])
#
#     dM = true_num_x + true_num_y - \
#          (num_model_correct_x + num_model_correct_y) + \
#          (num_model_wrong_x + num_model_wrong_y)
#     return dM
#
# # print(f" RE: {round(REbacres(x,y), 2)} | delta M: {dM}")
# print(f"delta M: {dM()}")



##

main_path = 'results/check_pf/'
sys_name = 'myvdp'
data_len = 'allonger'
snr = 'inf'
init = 0

# termsx = "#x: 1, x, xy, x^2"
# termsy = "#y: 1, xy, x^2"
# termsx = "# x: y"
# termsy = "# y: yx^2, x, y"
termss = [
    "# x: y",
    "# y: yx^2, x, y",
]

#glider
termss = [
    "# x: x^2, sin(y)",
    "# y: x, cos(y)/x",
]



print('starting loop')
def x_for_dM(termss=termss, sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path):
    """Extracts x and y from sindy results for dM metrics."""

    method = 'sindy'
    leng = 2000 if data_len == 'allonger' else 100
    path = f'{main_path}{sys_name}{os.sep}{method}{os.sep}{leng}{os.sep}'
    # file = f'{path}pf_prettytable_{data_len}_s0_e1_a2_len{leng}_snr{snr}_init{init}_{system}.csv'
    if method in ('dso', 'proged'):
        cases = f'len{leng}_snr{snr}_init{init}'
        pre = "pf"
    else:
        cases = ""
        pre = "overall_results"
    file = f'{path}{pre}_table_{data_len}_s0_e2_a1{cases}_{sys_name}.csv'

    # f = open(file, 'rb')
    # print(f.read())
    # f.close()
    # os.getcwd()
    # a = os.path.isfile(file)
    # # os.listdir(path)
    # b = os.listdir()
    # d = os.listdir(path)
    # print(file)
    #
    # c = os.path.isdir(path)
    # print(a, c, d )
    # print(b)
    csv = pd.read_csv(file, sep='\t')
    # def re(dx, dx_, dy, dy_):
    sys_name = csv['system'][0]

    re_reserved = ['^', '(', ')', '*', '+', '**']
    from sindy_lib import libd
    # sys_name = 'myvdp'
    lib = libd[sys_name]
    print(lib)
    def rewrap(word, problematic=re_reserved):
        for symbol in problematic:
            word = word.replace(symbol, f'\\{symbol}')
        return word
    # print(rewrap(dx, ['^', '(', ')']), 'rewrap')

    lib_re = [rewrap(word, problematic=re_reserved) for word in lib]

    dMs = []
    xys = []

    # print(csv.columns)
    systems = csv['expr_model']
    # systems = systems[:1]

    print()
    print('----- START HERE -----')
    print()
    for system in systems:
        xy = []
        dM = []
        print(system, 'system')
        if not sys_name == 'lorenz':
            exprs = re.findall(r'\'(.+)\', \'(.+)\'', system)
        else:
            exprs = re.findall(r'\'(.+)\', \'(.+)\', \'(.+)\'', system)
        dxdy = [expr + " " for expr in exprs[0]]
        for n, dx in enumerate(dxdy):
            bij = dict()
            for i in lib_re:
                regex = re.findall(r'(-?\d*\.?\d*) ' + i + ' ', dx)
                bij[i] = 0 if regex == [] else np.float64(regex[0])
            print(bij, 'bij')
            terms = termss[n]
            true_terms = [i.strip(' ') for i in terms.split(':')[1].split(',')]

            # print(true_terms, 'true_terms')
            # print('x^2' in true_terms, 'x^2 inside')
            # print(bij.get('x\^2', 404))
            true_terms_rewraped = [rewrap(word, problematic=re_reserved) for word in true_terms]
            x_correct = [bij.get(term, 0) for term in true_terms_rewraped]

            terms_correct = [term for term in true_terms_rewraped if term in bij]
            x_wrong = [bij[key] for key in bij if not key in true_terms and np.abs(bij[key]) > 0]
            xy += [(x_correct, x_wrong)]
            print('x:', x_correct, '\nexplain:', 'true_terms:', true_terms, 'terms_correct:', terms_correct, 'predicted expr:', dx)
            print(' '*50 + 'predicted terms:', [str(key) for key in bij])


            true_num_x = len(true_terms_rewraped)
            num_correct_x = len(np.nonzero(x_correct)[0])
            dM_x = true_num_x - num_correct_x + len(x_wrong)
            # dM = true_num_x + true_num_y - \
            #      (num_model_correct_x + num_model_correct_y) + \
            #      (num_model_wrong_x + num_model_wrong_y)
            print(true_num_x, num_correct_x, len(x_wrong))
            print(dM_x)
            dM += [dM_x]
        print(sum(dM))
        print(dM)
        xys += [xy]
        dMs += [sum(dM)]

    # pd.DataFrame(dMs)
    csv['dM'] = dMs
    # output = pd.concat([csv, df], axis=1)
    csv.to_csv(file[:-4] + '_dMs.csv', index=False)

    return xys, dMs

# print('x_for_dM', x_for_dM(sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path))

xys, dMs = x_for_dM(sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path)
print('dMs', dMs[:5])

from sindy_lib import libd

for sys_name in libd:
    for data_len in ['all', 'allonger']:
        x_for_dM(sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path)

main_path = 'results/check_pf/'
sys_name = 'myvdp'
data_len = 'allonger'
snr = 'inf'
init = 0

# termsx = "#x: 1, x, xy, x^2"
# termsy = "#y: 1, xy, x^2"
# termsx = "# x: y"
# termsy = "# y: yx^2, x, y"
termss = [
    "# x: y",
    "# y: yx^2, x, y",
]



# 1/0


# # # print(a['expr'])
# # print()
# # print(type(a[0]))
# # print(a[1])
# a = "[['1.000 y', '-1.000 x + 1.996 y + -1.996 x^2*y']]"
# b = "[['0.981 y + 0.002 y^3', '-0.993 x + 1.946 y + 0.006 y^2 + 0.003 y^3 + -1.955 x^2*y']]"
# #
# # true = [['y'], ['x^2*y', 'x', 'y']]
# import re
# if not system == 'lorenz':
#     reg = re.findall(r'\'(.+)\', \'(.+)\'', b)
# else:
#     reg = re.findall(r'\'(.+)\', \'(.+)\', \'(.+)\'', b)
# print(reg[0])
# dx = reg[0][1] + " "
# print(dx)
# reg = re.findall(r'\'(.+)\', \'(.+)\', \'(.+)\'', b)
# # reg
#
# ##
# # lib = ['x', 'y', 'x^2', 'y^2', 'x^3', 'y^3', '1']
# from sindy_lib import libd
# # sys_name = 'myvdp'
# lib = libd[sys_name]
# print(lib)
#
# # from inspect import signature
# #
# #
# # from sindy_lib import library
# # lambdas, vars = library('myvdp')
# # libo = []
# # for var in vars:
# #
# #     print([nargs(f) for f in lambdas])
# #     if
# #      len(signature(f).parameters)
# #     libo += [l(var) for l in lambdas]
# # print(libo)
# # 1/0
#
#
#
# re_reserved = ['^', '(', ')', '*', '+', '**']
# def rewrap(word, problematic):
#     for symbol in problematic:
#         word = word.replace(symbol, f'\\{symbol}')
#     return word
# # print(rewrap(dx, ['^', '(', ')']), 'rewrap')
#
# lib_re = [rewrap(word, problematic=re_reserved) for word in lib]
# for i in lib_re:
#     print(i)
#
# bij = dict()
# for i in lib_re:
#     regex = re.findall(r'(-?\d*\.?\d*) ' + i + ' ', dx)
#     bij[i] = 0 if regex == [] else np.float64(regex[0])
#     # \'(.+)\'', \'(.+)\', \'(.+)\'', b)
# print(dx)
# print(bij)
# # i = 'y^2'
# # b = re.findall(r'(-?\d*\.?\d*) ' + i + ' ', dx)
# # print(b)
#
# # x: 1, x, xy, x^2
# # y: 1, xy, x^2
# termsx = "#x: 1, x, xy, x^2"
# termsy = "#y: 1, xy, x^2"
# true_terms = [i.strip(' ') for i in termsx.split(':')[1].split(',')]
# print(true_terms)
# x_correct = [bij.get(term, 0) for term in true_terms]
# x_wrong = [bij[key] for key in bij if not key in true_terms]
# print(x_correct)
# 1/0
#
# print(REbacres([20, -1, -1, 0.5, 1], [10, -1, 0.5, 1]))
#
# # explain: true_terms: ['1', 'xy', 'x^2'] terms_correct: ['1'] predicted expr: 1.237 1 + 0.316 x + 1.201 y + -0.415 x^2 + -0.175 y^2 + 0.220 x*y + -0.311 x^3 + 0.050 y^3 + -0.984 x^2*y + -0.256 y^2*x
#
