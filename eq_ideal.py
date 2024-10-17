import re

def ideal_to_eqs(ideal: str, complexity: int = 10, verbosity=1) -> list:
    """
    Function to convert ideal to parsimony equations, which can be checked for correctness.

    Args:
        - ideal (list): list of generators produced by MB algorithm.

    Returns: list of equations, as candidates for equations discovery task.
    """

    generators_string = ideal[6:]
    if verbosity > 0:
        print('generators_string')
        print(generators_string)
        print(generators_string.replace(' ', ''))
    # gens = generators_string.replace(' ', '').split(',')
    gens = generators_string.split(',')

    eqs = []
    for gen in gens:
        if 'j' in gen:
            raise ValueError('Repair split(\',  \'), since split from ideal to generator failed!!!.')
        print(f'gen:{gen}')
        summands = [i for i in gen.split(' ')]
        summands = [j for j in [i.replace(' ', '') for i in summands] if j != '']  # clean-up empty strings.

        if verbosity > 0:
            for sumand in summands:
                print('   ', sumand)
            print(len(summands))
        if len(summands) <= complexity:
            eqs += [summands]

    print(gens)
    eqs = sorted(eqs, key=lambda x: len(x))

    # eqs = gens
    for i, eq in enumerate(eqs):
        print(f'len(eq {i}): {len(eq)}, eq {i}: {eq}')

    return eqs

def bitsizef(equation: list[str]):
    """Function determining the "bitsize" of an equation.
        e.g. ['(1200/4345)*n*a(n-2)', 'a(n-1)'] \-> (4+4 + 0)/2 = 4

    Args:
        - eq (list): list of strings, representing the equation, result of ideal_to_eqs function.

    Returns: int, representing the "bitsize" of the equation.
        I.e. bitsize([coefficient*monomial for summand in equation]) := sum([bitsize(summand) for summand in equation])/(#summands),
        where bitsize(coefficient) = (magnitude(nominator) + magnitude(denominator).

    """

    print('in bitsizef')

    def bitsize_summand(summand: str):
        """Returns the sum of number of digits of the numerator and denomumerator,
        e.g. '+(-1200/4345)*n*a(n-2)(' \-> 4+4 = 8
        """

        print('in bitsizef_summand')
        summand = ['a(n)', '-2*a(n-1)', 'n^4', '+(-7/16875)*a(n-1)^4', '-93090916800*n^2*a(n-1)', 'tm2*a' ][5]
        print(summand)
        coef = re.findall(r'^\+?\-?\(?\-?(\d+)/?(\d*)\)?', summand)
        if coef == []:
        #     # if re.findall(r'[a-zA-Z]', coef[0]) == []: return 0 else: raise ValueError('Variable have strange name or bug in code!!!.')
            bits = 0
        # elif:
        else:
            # return coef[0]
            nom, denom = coef[0]
            bits = len(nom)+len(denom)

        print(coef)
        print('return:', bits)
        print('punchline')

        1/0
        coefficient, monomial = summand.split('*')

        def bitsize_coefficient(coefficient: str):
            """Function determining the "bitsize" of a coefficient.
                e.g. '1200/4345' \-> (4+4)/2 = 4

            Args:
                - coefficient (str): string, representing the coefficient.

            Returns: int, representing the "bitsize" of the coefficient.
                I.e. bitsize(coefficient) := (magnitude(nominator) + magnitude(denominator))/2.

            """

            nominator, denominator = coefficient.split('/')
            return (len(nominator) + len(denominator))

        coefficient, monomial = summand.split('*')
        return (bitsize_coefficient(coefficient) + len(monomial))/2


    print(bitsize_summand(equation[1]))
    1/0
    bsize = sum([bitsize_summand(summand) for summand in equation])



    return


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
    ideal = res[2]

    print('po res')
    print(ideal)
    eqs = ideal_to_eqs(ideal, complexity=16)
    print('eqs:', eqs)
    bitsize = bitsizef(eqs[1])
    print(bitsize)
    1/0

    print('\n ------ after eqs ------')
    for eq in eqs:
        print(eq)
    # 1/0
    print('after error')

    # increasing_mb(seq_id, csv, max_order=2, n_more_terms=10, library='n', n_of_terms=2000)

