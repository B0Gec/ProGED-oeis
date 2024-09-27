import os
import subprocess
from cocoa_location import cocoa_location
import re

# def mb_oeis(points, )
EXECUTE_CMD = False
# EXECUTE_CMD = True


# most common polynomial variables indeterminates:
vars_default = 'x, y, z, t, u, v, w, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t'.split(', ')
# # print(vars_lib)
def vars_str(dim, visual='djus'):
    vars_dict = {
        # 'djus': ",".join(vars_default[:dim]), # [the uss]ual suspects
        # 'num': ",".join([f'x_{i}' for i in range(dim)]),
        # 'oeis': ",".join([f'a_{{n-{i}}}' for i in range(dim)]),
        'djus': vars_default[:dim], # [the uss]ual suspects
        'num':  [f'x_{i}' for i in range(dim)],
        'oeis': [f'a_{{n-{i}}}' for i in range(dim)],
        }

    vars_dict = { key: ",".join(vars) for key, vars in vars_dict.items()}
    # print(vars_dict)
    return vars_dict[visual]


def mb(points: list):
    """Runs 5 lines of cocoa code with given points and returns the ideal
        It makes sure the numper of variables is appropriate.
        Cocoa code:
            use P ::= QQ[x,y];
            ////y = x^2: success:
            Points := mat([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81]]);
            I := IdealOfPoints(P, Points);
            I;

    Console command:
        echo "Points := ... IdealOfPoints;I;" | ./CocoAInterpter
    """
    # print(points)
    dims = [len(p) for p in points]
    if len(set(dims)) != 1:
        raise ValueError('All points must have the same number of dimensions!!')
    dim = dims[0]

    # [f"x_{i}" for i in range(dim)]
    # vars = ",".join([f"x_{i}" for i in range(dim)])
    vars = vars_str(dim, 'djus')
    # print(vars)

    points
    # use P ::= QQ[x,y];
    cocoa_code = f"use P ::= QQ[{vars}];"
    cocoa_code += f"Points := mat({points});"
    cocoa_code += "I := IdealOfPoints(P, Points);I;"
    print('cocoa_code pretty printed:', cocoa_code.replace(';', ';\n'))
    # print('cocoa_code oneliner:\n', cocoa_code)

    # a) set location of cocoa file that will be executed by cocoa interpreter
    # print('Current linux bash location:')
    # print(os.getcwd())

    # b) execute cocoa file
    command = f"cd ../{cocoa_location[2:]}; echo \"{cocoa_code}\" | ./CoCoAInterpreter"
    # print(command)
    if EXECUTE_CMD:
        print("Executing LINUX command for real...")
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        # print("Command output: \n", output.decode())

        # b) extract output ignoring logo
        cocoa_res = re.findall(r'ideal.*\n', output.decode())[0][:-1]
        print(cocoa_res)

        first_generator = cocoa_res[6:].split(',')[0]
        print(first_generator)
        return first_generator, cocoa_res
    print("NOT Executing LINUX command for real... just simulating command")

    return

print(mb([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81]]))



def mb_wrap_old(filename = 'runable.cocoa5', file_dir = 'julia/mb/'):

    # a) set location of cocoa file that will be executed by cocoa interpreter
    print('Current linux bash location:')
    print(os.getcwd())
    filename_loc = os.getcwd() + file_dir
    # filename = 'runable.cocoa5'
    filefull = filename_loc + filename
    print(' -=- exiting Python... '*20)


    # b) execute cocoa file
    command = f"cd ../{cocoa_location[2:]}; ./CoCoAInterpreter {filefull}"
    # print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    # print("Command output: \n", output.decode())

    # b) extract output ignoring logo
    cocoa_res = re.findall(r'ideal.*\n', output.decode())[0][:-1]
    print(cocoa_res)

    first_generator = cocoa_res[6:].split(',')[0]
    print(first_generator)
