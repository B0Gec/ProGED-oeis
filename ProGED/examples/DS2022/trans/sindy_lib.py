
def library(sys_name):

    # library

    #  1. Initialize custom SINDy library up to sixth order polynomials

    # 1.1 library polynomial
    if sys_name in ['lv', 'vdp', 'myvdp', 'stl', 'lorenz', 'vdpvdpUS', 'stlvdpBL']:
        library_function_names = [lambda x: x,
                                  lambda x: x + '^2',
                                  lambda x, y: x + '*' + y,
                                  lambda x: x + '^3',
                                  lambda x, y: x + '^2' + '*' + y,
                                  lambda x, y: y + '^2' + '*' + x,
                                  ]
        if sys_name == 'lorenz':
            var_names = ["x", "y", "z"]
        elif sys_name in ['vdpvdpUS', 'stlvdpBL']:
            var_names = ["x", "y", "u", "v"]
        else:
            var_names = ["x", "y"]

    elif sys_name in ['bacres', 'glider', 'predprey', 'shearflow']:

        library_function_names = [lambda x: x,
                                  lambda x: x + '^2',
                                  lambda x: x + '^3',
                                  lambda x, y: x + '*' + y,
                                  lambda x, y: x + '/' + y,
                                  lambda x, y: y + '/' + x,
                                  lambda x, y: x + '^2' + '*' + y,
                                  lambda x, y: y + '^2' + '*' + x,

                                  lambda x: 'sin(' + x + ')',
                                  lambda x: 'cos(' + x + ')',
                                  lambda x: 'tan(' + x + ')',
                                  lambda x: 'cot(' + x + ')',

                                  lambda x, y: 'cos(' + y + ')/(' + x + ')',
                                  lambda x, y: 'sin(' + y + ')/(' + x + ')',
                                  lambda x, y: 'cos(' + x + ')/(' + y + ')',
                                  lambda x, y: 'sin(' + x + ')/(' + y + ')',

                                  lambda x, y: 'sin^2(' + y + ') * sin(' + x + ')',
                                  lambda x, y: 'sin^2(' + x + ') * sin(' + y + ')',
                                  lambda x, y: 'cos^2(' + x + ') * sin(' + y + ')',
                                  lambda x, y: 'cos^2(' + y + ') * sin(' + x + ')',
                                  lambda x, y: 'sin(' + x + ') * cot(' + y + ')',
                                  lambda x, y: 'sin(' + y + ') * cot(' + x + ')',
                                  lambda x, y: 'cos(' + x + ') * cot(' + y + ')',
                                  lambda x, y: 'cos(' + y + ') * cot(' + x + ')',
                                  ]

        var_names = ["x", "y"]
        # feature names: ['1', 'x', 'y', 'x^2', 'y^2', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'xy', 'x/y', 'y/x', 'cos(y)/(x)', 'x^3', 'y^3', 'x^2y', 'y^2x']

    elif sys_name in ['barmag', 'cphase']:

        library_function_names = [lambda x: 'sin(' + x + ')',
                                  lambda x: 'cos(' + x + ')',
                                  lambda x, y: 'sin(' + x + '+' + y + ')',
                                  lambda x, y: 'cos(' + x + '+' + y + ')',
                                  lambda x, y: 'sin(' + x + '-' + y + ')',
                                  lambda x, y: 'cos(' + x + '-' + y + ')',
                                  lambda x, y: 'sin(' + y + '-' + x + ')',
                                  lambda x, y: 'cos(' + y + '-' + x + ')',
                                  ]
        if sys_name == 'cphase':
            var_names = ["x", "y", "t"]
        else:
            var_names = ["x", "y"]
            # ['1', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'sin(x+y)', 'sin(x-y)', '(sin(x)^2 + cos(x)^2)*sin(y)', 'cos(x) cot(y)']    else:
    else:
        print("Error. No library could be chosen based on the system name (sys_name). Recheck inputs.")

    return library_function_names, var_names

libd = {
'bacres': ['1', 'x', 'y', 'x^2', 'y^2', 'x^3', 'y^3', 'x*y', 'x/y', 'y/x', 'x^2*y', 'y^2*x', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'cos(y)/(x)', 'sin(y)/(x)', 'cos(x)/(y)', 'sin(x)/(y)', 'sin^2(y) * sin(x)', 'sin^2(x) * sin(y)', 'cos^2(x) * sin(y)', 'cos^2(y) * sin(x)', 'sin(x) * cot(y)', 'sin(y) * cot(x)', 'cos(x) * cot(y)', 'cos(y) * cot(x)'],
'barmag': ['1', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'sin(x+y)', 'cos(x+y)', 'sin(x-y)', 'cos(x-y)', 'sin(y-x)', 'cos(y-x)'],
'glider': ['1', 'x', 'y', 'x^2', 'y^2', 'x^3', 'y^3', 'x*y', 'x/y', 'y/x', 'x^2*y', 'y^2*x', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'cos(y)/(x)', 'sin(y)/(x)', 'cos(x)/(y)', 'sin(x)/(y)', 'sin^2(y) * sin(x)', 'sin^2(x) * sin(y)', 'cos^2(x) * sin(y)', 'cos^2(y) * sin(x)', 'sin(x) * cot(y)', 'sin(y) * cot(x)', 'cos(x) * cot(y)', 'cos(y) * cot(x)'],
'lv': ['1', 'x', 'y', 'x^2', 'y^2', 'x*y', 'x^3', 'y^3', 'x^2*y', 'y^2*x'],
'predprey': ['1', 'x', 'y', 'x^2', 'y^2', 'x^3', 'y^3', 'x*y', 'x/y', 'y/x', 'x^2*y', 'y^2*x', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'cos(y)/(x)', 'sin(y)/(x)', 'cos(x)/(y)', 'sin(x)/(y)', 'sin^2(y) * sin(x)', 'sin^2(x) * sin(y)', 'cos^2(x) * sin(y)', 'cos^2(y) * sin(x)', 'sin(x) * cot(y)', 'sin(y) * cot(x)', 'cos(x) * cot(y)', 'cos(y) * cot(x)'],
'shearflow': ['1', 'x', 'y', 'x^2', 'y^2', 'x^3',  'y^3', 'x*y', 'x/y', 'y/x', 'x^2*y', 'y^2*x', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'cos^2(y)', 'cos^2(x)', 'sin^2(y)', 'sin^2(x)', 'cos(y)/(x)', 'sin(y)/(x)', 'cos(x)/(y)', 'sin(x)/(y)', 'sin^2(y) * sin(x)', 'sin^2(x) * sin(y)', 'cos^2(x) * sin(y)', 'cos^2(y) * sin(x)', 'sin(x) * cot(y)', 'sin(y) * cot(x)', 'cos(x) * cot(y)', 'cos(y) * cot(x)'],
# 'shearflow': ['1', 'x', 'y', 'x^2', 'y^2', 'x^3', 'y^3', 'xy', 'x/y', 'y/x', 'x^2y', 'y^2x', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'tan(x)', 'tan(y)', 'cot(x)', 'cot(y)', 'sin^2(x)', 'sin^2(y)', 'cos^2(x)', 'cos^2(y)', 'cos(y)/(x)', 'sin(y)/(x)', 'cos(x)/(y)', 'sin(x)/(y)', 'sin^2(y) sin(x)', 'sin^2(x) * sin(y)', 'cos^2(x) * sin(y)', 'cos^2(y) * sin(x)', 'sin(x) * cot(y)', 'sin(y) * cot(x)', 'cos(x) * cot(y)', 'cos(y) * cot(x)'],
# 'vdp': ['1', 'x', 'y', 'x^2', 'y^2', 'x*y', 'x^3', 'y^3', 'x^2*y', 'y^2*x'],
'myvdp': ['1', 'x', 'y', 'x^2', 'y^2', 'x*y', 'x^3', 'y^3', 'x^2*y', 'y^2*x'],
'stl': ['1', 'x', 'y', 'x^2', 'y^2', 'x*y', 'x^3', 'y^3', 'x^2*y', 'y^2*x'],
'cphase': ['1', 'sin(x)', 'sin(y)', 'sin(t)', 'cos(x)', 'cos(y)', 'cos(t)', 'sin(x+y)', 'sin(x+t)', 'sin(y+t)', 'cos(x+y)', 'cos(x+t)', 'cos(y+t)', 'sin(x-y)', 'sin(x-t)', 'sin(y-t)', 'cos(x-y)', 'cos(x-t)', 'cos(y-t)', 'sin(y-x)', 'sin(t-x)', 'sin(t-y)', 'cos(y-x)', 'cos(t-x)', 'cos(t-y)'],
'lorenz': ['1', 'x', 'y', 'z', 'x^2', 'y^2', 'z^2', 'x*y', 'x*z', 'y*z', 'x^3', 'y^3', 'z^3', 'x^2*y', 'x^2*z', 'y^2*z', 'y^2*x', 'z^2*x', 'z^2*y'],
}



