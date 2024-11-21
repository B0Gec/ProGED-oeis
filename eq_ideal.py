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


def linear_to_vec(linear_eq):
    return

def check_implicit(mb_eq, order):
    """ I checked that I have not already implemented this function in exact_ed.py or check_quick.py.
    Although check_eq_man has similar implementation I will try to implement it from the scratch. 
    """
    
    eq = 'a(n) -2*a(n-1)'

    eq = 'n*a(n-1)^3 +(-7/16875)*a(n-1)^4'

    from exact_ed import var2term_v2, stvar2term_v2
    print('stvar2term', stvar2term_v2(stvar='n', till_now=[0, 1, 1, 2, 3, 5], order_=1))
    print('var2term', var2term_v2(var='n*a(n-1)^3', till_now=[0, 1, 1, 2, 3, 5], order_=1))
    1/0

    return

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
    from exact_ed import unnan

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
    check_implicit(eqs[1], 1)
    1/0

    print('\n ------ after eqs ------')
    for eq in eqs:
        print(eq)
    # 1/0
    print('after error')

    # increasing_mb(seq_id, csv, max_order=2, n_more_terms=10, library='n', n_of_terms=2000)

