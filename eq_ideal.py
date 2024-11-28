"""
This file tries to automate the translation from ideal to equation.
I.e., given output of MB algorithm, find the most appropriate equation automatically.
Also, take care of _linrec_ experiments.

Strategy:
    - sort generators of the ideal by complexity and bitsize. Bitsize of a polynomial expression is the sum of bitsizes of
     all summands. For equations of bitsize <= 20, sort by number of summands, otherwise by bitsize.
    - return top_n equations and filter by max_complexity and max_bitsize, output only if not too big numbers involved
    or number of summands.
    - Linrec: limit complexity to 20 and make an additional list of purely linear equations.
        - create 2 functions for automatic verification:
            1.) linear expression: convert easily to recursive linear/vector form for ground check. | -> id
            2.) implicit expression: function to verify it on the whole sequence. | -> non_id (and man_check)
    - TM-OEIS:
            3.) implicit to explicit: to predict next terms for TM-OEIS benchmark.
            - Run experiments for SINDy to check if it is better than TM-OEIS (fun if seen better than transformers).
"""



import re


def bitsize_summand(summand: str):
    """Returns the sum of number of digits of the numerator and denomumerator,
    e.g. '+(-1200/4345)*n*a(n-2)(' \-> 4+4 = 8
    """

    # print('in bitsizef_summand')
    # summand = ['a(n)', '-2*a(n-1)', 'n^4', '+(-7/16875)*a(n-1)^4', '-93090916800*n^2*a(n-1)', 'tm2*a' ][5]
    # print(summand)
    coef = re.findall(r'^\+?\-?\(?\-?(\d+)/?(\d*)\)?', summand)
    if coef == []:
        #     # if re.findall(r'[a-zA-Z]', coef[0]) == []: return 0 else: raise ValueError('Variable have strange name or bug in code!!!.')
        bits = 1
    # elif:
    else:
        # return coef[0]
        nom, denom = coef[0]
        bits = len(nom) + len(denom)
        # print('coef', f'{str(coef): <40}, bits: {len(nom)} + {len(denom)} = {bits}', 'summand:', summand)

    if len(coef) > 1:
        raise ValueError('Strange error in bitsize_summand function!!!      ... input has more than one coefficient!!!')
    # print(coef)
    # print('return:', bits)
    # print('punchline')

    return bits


def bitsize(equation: list[str]):
    """Function determining the "bitsize" of an equation.
        e.g. ['(1200/4345)*n*a(n-2)', 'a(n-1)'] \-> (4+4 + 0)/2 = 4

    Args:
        - eq (list): list of strings, representing the equation, result of ideal_to_eqs function.

    Returns: int, representing the "bitsize" of the equation.
        I.e. bitsize([coefficient*monomial for summand in equation]) := sum([bitsize(summand) for summand in equation])/(#summands),
        where bitsize(coefficient) = (magnitude(nominator) + magnitude(denominator).

    """

    # print('in bitsizef')

    # print(bitsize_summand(equation[1]))
    bsize = sum([bitsize_summand(summand) for summand in equation])
    # print('eq:', equation)
    # print(f'returning bitsizef({equation}):\n', bsize)
    # for i in equation:
    #     print('  ', i, bitsize_summand(i))

    return bsize


def ideal_to_eqs(ideal: str, max_complexity: int = 10, max_bitsize: int = 100, top_n: int = 10, verbosity=1) -> list:
    """
    Function to convert ideal to parsimony equations, which can be checked for correctness.

    Args:
        - ideal (list): list of generators produced by MB algorithm.

    Returns: list of equations, as candidates for equation discovery task.

    Filters:
        - max_complexity (int): Only equations with complexity <= max_complexity summands are outputed.
        - max_bitsize (int): Only equations with bitsize <= max_bitsize summands are outputed.
    """

    generators_string = ideal[6:]
    if verbosity >= 2:
        print('generators_string')
        print(generators_string)
        print(generators_string.replace(' ', ''))
    gens = generators_string.split(',')

    eqs = []
    for gen in gens:
        if 'j' in gen:  # wtf!?
            raise ValueError('Repair split(\',  \'), since split from ideal to generator failed!!!.')
        # print(f'gen:{gen}')
        summands = [i for i in gen.split(' ')]
        summands = [j for j in [i.replace(' ', '') for i in summands] if j != '']  # clean-up empty strings.

        if verbosity >=2:
            for sumand in summands:
                print('   ', sumand)
            print(len(summands))
        if len(summands) <= max_complexity and bitsize(summands) <= max_bitsize:  # filter equations by complexity
            eqs += [summands]

    # 1/0
    # eqs0 = sorted(eqs, key=lambda x: (bitsize(x), len(x)))[:top_n//2]
    eqs0 = sorted(eqs, key=lambda x: (len(x), bitsize(x)) if bitsize(x) <= 20 else (bitsize(x), len(x)))[:top_n]
    # x + y - z ... x + 100*y  now: 0 vs 3 (or updated 3 vs 4)
    # 10x + 34y - 5z ... 214x + 100*y  5 vs 6.
    # eqs = eqs0 + [i for i in sorted(eqs, key=lambda x: (len(x), bitsize(x))) if i not in eqs0][:top_n//2]
    eqs1 = [i for i in sorted(eqs, key=lambda x: (len(x), bitsize(x))) if i not in eqs0][:top_n//2]
    # eqs = eqs0 + eqs1
    eqs = eqs0

    if verbosity >= 1:
        for i, eq in enumerate(eqs):
            print(f'len(eq {i}): {len(eq)}, eq {i}: {eq}')

    human_readable_eqs = [' '.join(eq) for eq in eqs]
    # print('human_readable_eqs', human_readable_eqs)

    return eqs, human_readable_eqs

def is_linear(expr: str) -> bool:
    """Check if the OEIS recursive expression is linear without constant terms.
        Used by Linrec experiments for linear_to_vec function.
    """

    if '^' in expr:  # Power to > 1 indicates nonlinearity.
        return False
    else:  # Check if there are degree >= 2 terms.
        # noty = re.findall(r'^\+?-?\(?-?\d+/?\d*\)?\*?[^ ]+\*[^ ]+', expr)
        # noty = re.findall(r'[an][(n\-0-9)]*\*', expr) # this is enough here
        quads = re.findall(r'[an][(n\-0-9)]*\*[an][(n\-0-9)]*', expr)  # for linear including 'n'
        if quads:
            return False
        else:  # check if constant term is present
            summands = [i for i in expr.split(' ')]
            summands = [j for j in [i.replace(' ', '') for i in summands] if j != '']  # clean-up empty strings.
            # print('summands:', summands)
            constants = sum([re.findall('^[^an*]*$', summand) for summand in summands], [])
            return len(constants) == 0

def order_optimize(expr: str) -> str:
    """Preprocess the expression (not containing the 'n' variable,) to minimize the recursion order.
        E.g. a(n-1) - 2*a(n-2) -> a(n) - 2*a(n-1)

    Useful for finding true vector for linear recursive equations of OEIS in Linrec data set.
        Careful! This function is not for general use:
            'a(n-2) - a(n-1) - n' \-> 'a(n-1) - a(n) - n' and not 'a(n) - a(n-1) - (n+1)'
    """

    min_order, max_order = eq_order_explicit(expr)
    # print(min_order, max_order)
    # print(expr)
    if not min_order in (None, 0):
        # a(n-1) -> a(n)
        expr = expr.replace(f'a(n-{min_order})', 'a(n)')
        # print('expr after a(n-1) -> a(n):', expr)
        # in the end: n-min -> n
        for i in range(min_order+1, max_order+1):
            # print('i:', i)
            expr = expr.replace(f'a(n-{i})', f'a(n-{i-min_order})')
            # print('expr after a(n-3) -> a(n-2):', expr)
    # print('expr after all:', expr)
    return expr


def linear_to_vec(linear_expr: str) -> list:
    """Convert linear expression from mb to vector form.
    E.g. 'a(n) - a(n-1) - a(n-2)' -> [1, 1]

    Details:
        Expression is equalized to zero, and then a(n) is isolated and divided by its coefficient.
        the vector of coefficients in the rhs is returned.
    Input:
        - expression that was checked to be linear.
    """

    from exact_ed import eq_order_explicit
    from mb_wrap import cocoa_eval
    from mb_oeis import pretty_to_cocoa

    # head_tail = expr.split('a(n)')
    # if len(head_tail) != 1:
    expr = 'a(n) -3*a(n-2) +a(n-2)'
    # expr = '+a(n) -3*a(n-2) +a(n-2)'
    # expr = '-2*a(n) -3*a(n-2) +a(n-2)'
    linear_expr = '3*a(n-3) -a(n) -3*a(n-2) +a(n-2)'
    # expr =   '+(-7/16875)*a(n) +(-200704/15)*n^3 '
    print('\nexpr:', linear_expr)
    expr = order_optimize(linear_expr)
    order = eq_order_explicit(expr)[1]

    summands = [i for i in expr.split(' ')]
    summands = [j for j in [i.replace(' ', '') for i in summands] if j != '']  # clean-up empty strings.
    lhs = [summand for summand in summands if 'a(n)' in summand]
    rhss = [summand for summand in summands if 'a(n)' not in summand]
    if len(lhs) != 1:
        print('a(n) monomials:', lhs)
        raise ValueError('Strange error in linear_to_vec function!!!      '
                         '... linear expression has more than one \'a(n)\' (or not even one) monomials !!!'
                         f'\nThese are: {lhs}.')
    print("lhs, rhs:", lhs, rhss)
    print('summands', summands)
    coef = re.findall(r'^\+?([^an*]*)\*?a\(n\)', lhs[0])[0]
    cases = {'': '1', '+': '1', '-': '-1'}
    coef = cases[coef] if coef in list(cases.keys()) else coef
    print('coef', coef)
    rhs = "".join(rhss)
    rhs = pretty_to_cocoa(rhs, order)
    print('rhs', rhs)

    vars = ",".join(['a(n)'] + [f'a(n-{i})' for i in range(order+1)])
    print(vars)
    preamble = f'P::= QQ[{vars}];'
    # divided = cocoa_eval(preamble + f'({rhs})/({coef});', execute_cmd=True, verbosity=3, cluster=False)
    # rhs = divided
    print('rhs', rhs)

    return

def check_implicit(mb_eq: str, seq: list[int]) -> bool:
    """ I checked that I have not yet implemented this function in exact_ed.py or check_quick.py.
    Although check_eq_man has similar implementation I implemented it from the scratch with hope
    of more efficiency by using Cocoa.

    E.g. 'a(n) - a(n-2) - 1*a(n-1)^1', [0, 1, 1, 2, 3, 5] -> True
    """

    from exact_ed import expr_eval, obs_eval, eq_order_explicit

    order = eq_order_explicit(mb_eq)[1]
    wanted_zeros = []
    for n in range(order, len(seq)):
        till_now = seq[:n+1]
        # print('n', n, 'till_now', till_now)
        evaled = expr_eval(mb_eq, till_now)
        # print('n', n, 'till_now', till_now, f'evaled: {evaled}')
        wanted_zeros.append(evaled)
    # print(wanted_zeros)
    non_zeros = [i for i in wanted_zeros if i != '0']
    vanishes = len(non_zeros) == 0
    return vanishes

# maybe for TM-OEIS:
# def implicit_to_explicit(linear_eq):
#     return


if __name__ == '__main__':
    outpt = 'a_n*a_n_1 +a_n*a_n_2 -a_n_1*a_n_3 -a_n_2*a_n_3 -3*a_n +3*a_n_3'
    outpt = 'a_n_1 * a_n_2 + a_n_1 * a_n_3 - a_n_2 * a_n_4 - a_n_3 * a_n_4 - 3 * a_n_1 + 3 * a_n_4'
    outpt = 'a_n * a_n_1 + a_n * a_n_2 + a_n_1 * a_n_2 - 3 * a_n - 3 * a_n_1 - 3 * a_n_2 + 7'

    # print(external_prettyprint(outpt, sol_ref= ['a(n-1)', 'a(n-2)', 'a(n-3)', 'a(n-4)', 'a(n-5)']))
    # 1/0

    import pandas as pd

    from mb_oeis import one_mb
    # import matplotlib.pyplot as plt
    # from mavi_oeis import anform

    eq = """
        0.01⋅a(n) - 0.004⋅a(n - 3) - 0.02⋅a(n - 1) + 0.02 = 0
        """
    # print(anform(eq, 0.01))
    # 1/0

    seq_id = 'A000045'
    seq_id = 'A000079'
    csv_filename = 'cores_test.csv'
    N_OF_TERMS_LOAD = 10 ** 5
    N_OF_TERMS_LOAD = 20
    # N_OF_TERMS_LOAD = 10
    # N_OF_TERMS_LOAD = 2
    csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])[:N_OF_TERMS_LOAD]

    from IPython.display import display, display_latex
    from exact_ed import unnan, eq_order_explicit

    seq, coeffs, truth = (unnan(csv[seq_id]), None, None)
    print(seq)
    # 1/0

    # # seq = sindy_hidden[d_max_lib - 1]
    # # print(n_of_terms, seq)
    # seq = seq[:N_OF_TERMS_LOAD]
    # print(seq)
    # # 1/0


    print('before onemb')
    res = one_mb(seq, order=1, n_more_terms=10, execute=True, library='n', n_of_terms=200)
    print(res)
    # 1/0
    ideal = res[2]

    print('po res')
    print(ideal)
    eqs, heqs = ideal_to_eqs(ideal, max_complexity=16, max_bitsize=100, top_n=10, verbosity=0)
    # eqs = ideal_to_eqs(ideal, complexity=16, top_n=10, verbosity=2)
    # for eq in eqs:
    #     print(f'  len(eq): {len(eq)}, bitsize: {bitsize(eq)}, bitsizes {[bitsize_summand(summand) for summand in eq]}, \neq: {eq}')

    print('eqs:', eqs)
    print('eqs:', len(eqs))
    bitsize_ = bitsize(eqs[1])
    print(bitsize_)

    expr = 'a(n) -a(n-1) -a(n-2)'
    expr = ' -a(n-1) -a(n-2)'
    # expr = 'n + n^2 - n^3'
    # expr = 'a(n-6)*a(n+14) - a(n-2)'
    # expr = 'a(n-6) +a(n-14) -a(n-2)'
    expr = 'a(n) +13*n -34'
    # expr =   '+(-7/16875)*a(n-1)^4 +(-200704/15)*n^3 '
    # expr =   '+(-7/16875)*a(n-1)*a(n-2) +(-200704/15)*n*a(n)'

    seq = [0, 1, 1, 2, 3, 5]
    print(expr, seq)
    order = eq_order_explicit(expr)
    print('eq_order:', order)
    # is_check = check_implicit(expr, seq)
    # print('check_implicit:', is_check)
    is_linear_ = is_linear(expr)
    print('is_linear_:', is_linear_)
    # 1/0
    order_optimized = order_optimize(expr)
    print('order_optimized:', order_optimized)

    expr = 'a(n) -a(n-1) -a(n-2)'
    print()
    print('ablated expr:', expr)
    vector = linear_to_vec(expr)
    print('vector:', vector)
    1/0

    print('\n ------ after eqs ------')
    for eq in eqs:
        print(eq)
    # 1/0
    print('after error')

    # increasing_mb(seq_id, csv, max_order=2, n_more_terms=10, library='n', n_of_terms=2000)

