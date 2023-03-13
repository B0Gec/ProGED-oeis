import os, sys
import re

dire = 'results/ph2/'
# print(os.path.isdir(dire))


files = []

buff = 5
headlast = 114
last = 4

# n, fn, s, max_iter, pop_size, model, error
# n, fn, s, max_iter, pop_size, model, error
d = dict()

for n, fn in enumerate(os.listdir(dire)[:1005]):
    print(n, fn)
    if fn[-4:] != '.txt':
        continue
    # if n>4:
    #     print('exit')
    #     break
    # if fn != 'phfit_4.txt':
    #     continue


    f = open(dire + fn, 'r') 
    # head = f.read(10**buff)[:10**last]
    # tail = f.read(10**buff)[-10**last:]
    whole = f.read()
    # print(head)
    # print(tail)
    # print(whole)
    # print(fn)
    # files += [(head, tail)]
    # files += [whole]
    f.close()
    # break

    # print()
    # print('re')
    # h = files[0]
    # h = files[0][0]
    h = whole
    # reg = re.findall(r'.+\n.+max_iter.+\n', h)
    reg = re.findall(r'.+\n.*s.+\d.+max_iter.+', h)
    # reg = re.findall(r'max_iter', h)
    # print(h)
    # reg = re.findall(r'max_iter', h)
    # print(h)
    # print('header, obs:\n\n')
    # print('\n'*3)
    # print(f'    - - - - - {fn} - - - - - - - - ')
    if len(reg) == 0:
        print('didnt find the max_iter')
        # print(h)
        # print('didnt find the max_iter')
        continue
        break
    else:
        # continue
        print(reg[0])
    # print('stop - this was header')
    # print()
    # print('after continue')

    # t = files[0][1]
    t = h
    # print(f)
    # reg = re.findall(r'\n.+\n.+\n.+\n.+pop_size.+\n.+\n.+\n.+', t)
    # reg = re.findall(r'((.*\n){0,5}.*\n.+pop_size.+\n.+\n\nEOF\n lsoda)', t)
    res = re.findall(r'(model.+)p.+(error.+)', t)
    # print(res[1])

    if len(res) == 0:
        print('didnt find the model and error')
    else:
        print(res[0][0] + res[0][1])
    # print(res)
    # print()
    # print(f'    \\____________ end of {fn} _________//// ')
    # print('\n'*3)
    # break

print(' --- ')
# # reg = re.findall(r'\n.+pop_size.+\n.+', f)
# print('regs:')
# for i in reg:
#     print(i[0])
#     # for j in i:
#     #     print('j:', j)
# # print('firstreg\n', reg[0])
# # reg = re.findall(r'.+False.+', f)
# # print(reg)
# # reg = re.findall(r'.+True.+', f)
# # print(reg)
# # rega = re.findall(r'.+popsize.+', 'dst dst dst dst popsize dstdstds')
# # print(reg, rega)





print(2)
