# helping calculating reconstruction error manually
import numpy as np
import pandas as pd
import os
import re


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

# my barmag, stl, mvdp
correct_systems = {
    'bacres': [
    ],
    'barmag': [
        "# x: sin(x-y), sin(x)",
        "# y: sin(y-x), sin(y)",
    ],
    'glider': [
        "# x: x^2, sin(y)",
        "# y: x, cos(y)/(x)",
    ],
    'lv': [
        "# x: x, xy, x^2",
        "# y: y, xy, y^2",
    ],
    'predprey': [
    ],
    'shearflow': [
        "# x: cos(x) * cot(y)",
        "# y: cos(y)^2, sin^2(y) * sin(x)",
    ],
    'myvdp': [
        "# x: y",
        "# y: yx^2, x, y",
    ],
    'stl': [
        "# x: x, y, x^3, y^2*x",
        "# x: x, y, x^2*y, y^3",
    ],
    'cphase': [
    ],
    'lorenz': [
        "# x: y, x",
        "# y: x, x*z, y",
        "# z: z, x*y",
    ],
}



print('starting loop')
def x_for_dM(correct_systems=correct_systems, sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path):
    """Extracts x and y from sindy results for dM metrics."""

    termss = correct_systems[sys_name]

    print()
    print('----- START x_for_dM HERE -----')
    print()
    print(termss, sys_name, data_len, snr, init)
    if termss == []:
        print('for this sistem we wont calculate delta M')
        return

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
    sea = ['s0', 'e2', 'a1',]
    combs = [ ['s0', i, j] for i in ['e1', 'e2',] for j in ['a1', 'a2',]]
    # sea = ['s0', 'e2', 'a1',]
    file_frame = f'{path}{pre}_table_{data_len}_{{}}_{{}}_{{}}{cases}_{sys_name}.csv'
    print(combs, file_frame)

    for c in combs:
        file = file_frame.format(c[0], c[1], c[2])
        print(file)
        if os.path.isfile(file):
            break

    print(file)
    # 1/0





    # f = open(file, 'rb')
    # print(f.read())
    # f.close()
    # os.getcwd()
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
        dxdy = [expr + " +" for expr in exprs[0]]
        print(dxdy)
        for n, dx in enumerate(dxdy):
            print(dx, 'dx')
            bij = dict()
            for i in lib_re:
                # print(i, 'i')
                # print(dx, 'dx')
                regex = re.findall(r'(-?\d*\.?\d+) ' + i + ' \+', dx)
                # regex = re.findall(r'(-?\d*\.?\d*) ' + 'sin\(y\)' + ' ', dx)
                # print(regex)
                bij[i] = 0 if regex == [] else np.float64(regex[0])
                # print(bij)
            print(bij, 'bij')
            terms = termss[n]
            true_terms = [i.strip(' ') for i in terms.split(':')[1].split(',')]

            print(true_terms, 'true_terms')
            # print('x^2' in true_terms, 'x^2 inside')
            # print(bij.get('x\^2', 404))
            true_terms_rewraped = [rewrap(word, problematic=re_reserved) for word in true_terms]
            x_correct = [bij.get(term, 0) for term in true_terms_rewraped]

            terms_correct = [term for term in true_terms_rewraped if term in bij and np.abs(bij[term]) > 0]
            x_wrong = [bij[key] for key in bij if not key in true_terms_rewraped and np.abs(bij[key]) > 0]
            terms_wrong = [key for key in bij if (not key in true_terms_rewraped) and np.abs(bij[key]) > 0]
            xy += [(x_correct, x_wrong)]
            print('x:', x_correct, '\nexplain:', 'true_terms:', true_terms_rewraped, 'terms_correct:', terms_correct, 'predicted expr:', dx)
            print(' '*50 + 'predicted terms:', [str(key) for key in bij if abs(bij[key])>0])
            print(' '*50 + 'wrong terms:', terms_wrong)


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
        # break
    # break
    csv['dM'] = dMs
    csv.to_csv(file[:-4] + '_dMs.csv', index=False)

    return xys, dMs

# print('x_for_dM', x_for_dM(sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path))

# xys, dMs = x_for_dM(sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path)
# print('dMs', dMs[:5])

from sindy_lib import libd

for sys_name in libd:
    for data_len in ['all', 'allonger']:
        x_for_dM(correct_systems=correct_systems, sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path)

main_path = 'results/check_pf/'
sys_name = 'myvdp'
sys_name = 'glider'
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
    "# y: x, cos(y)/(x)",
]
glider_file = "overall_results_table_all_s0_e1_a1_glider.csv"
# 1/0
# for data_len in ['all', 'allonger']:
# for data_len in ['allonger']:
#     xys, dms = x_for_dM(correct_systems=correct_systems, sys_name=sys_name, data_len=data_len, snr=snr, init=init, check_pf_path=main_path)
#     print(dms)



