import pandas as pd
import sympy as sp

from exact_ed import unpack_seq, truth2coeffs,solution_vs_truth, instant_solution_vs_truth, solution2str


csv_filename = '../linear_database_full.csv'
# if CORELIST:
#     # from core_nice_nomore import cores
#     csv_filename = 'cores.csv'
#
# # print(os.getcwd())
# if os.getcwd()[-11:] == 'ProGED_oeis':
#     # csv_filename = 'ProGED_oeis/examples/oeis/' + csv_filename
#     # print(os.getcwd())
#     pass
# # except ValueError as error:
# #     print(error.__repr__()[:1000], type(error))

seq_id = 'A000045'
# csv = pd.read_csv(csv_filename, low_memory=False, nrows=0)
csv = pd.read_csv(csv_filename, low_memory=False, usecols=[seq_id])
# print(csv)

print(unpack_seq(seq_id, csv))
print(unpack_seq(seq_id, csv)[1])
print(solution2str(sp.Matrix([1, 0, 1, 0, 0, 0, 1])))
print(solution2str([]))
# print(solution_vs_truth([]))
print(instant_solution_vs_truth(sp.Matrix([1, 0, 1, 0, 0, 0, 1]), seq_id, csv))
print(instant_solution_vs_truth(sp.Matrix([0, 1, 1, 0, 0, 0, 0]), seq_id, csv))
print(instant_solution_vs_truth(sp.Matrix([0, 1, 1, 0, 0, 0, 0]), seq_id, csv))
print(solution_vs_truth(sp.Matrix([0, 1, 1, 0, 0, 0, 0]), sp.Matrix([1, 1])))


def sindy(task_id: int, seq_id: str, csv: pd.DataFrame, now: int):
    # x, coeffs, eq, truth = exact_ed(seq_id, csv, VERBOSITY, MAX_ORDER,
    #                                 n_of_terms=N_OF_TERMS_ED, linear=True)

    # perform sindy
    # check against ground truth

    return



