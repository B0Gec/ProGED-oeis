"""
Generate unstructions for llm.
"""

import pandas as pd

from exact_ed import unpack_seq
# train_csv = pd.read_csv('linrec_without_dasco.csv', low_memory=False)
train_csv = pd.read_csv('linrec_without_dasco.csv', low_memory=False, usecols=['A000051'])

a = train_csv.columns[0]
print(a)
print(train_csv[a])
col = train_csv[a].dropna()
seq_matrix, coeffs_matrix, truth = unpack_seq(a, train_csv)
print()
# print(truth)
seq, coeffs = list(seq_matrix), list(coeffs_matrix)
truth = ['a_n = '] + [f'{str(coeff)}*a_{{n-{n + 1}}} + ' for n, coeff in enumerate(coeffs_matrix)]
eq = ''.join(truth)[:-3]
# print(truth)
print(eq)
print(seq)
print(','.join([str(i) for i in seq]))
# print(coeffs, coeffs_matrix)
# print(truth)
# 1/0


# seq =
print(col)
def sequence_to_parts(seq, length=15):
    """Separate a sequence into parts/chunks of given length."""
    # seq = [i for i in range(100)]
    return [seq[i:i+length] for i in range(0, len(seq), length)]

print(sequence_to_parts(seq, 15))

def column_to_prompts(col_id):
    """Convert a column of a dataframe to multiple prompts."""
    seq_matrix, coeffs_matrix, truth = unpack_seq(col_id, train_csv)
    seq = list(seq_matrix)
    truth = ['a_n = '] + [f'{str(coeff)}*a_{{n-{n + 1}}} + ' for n, coeff in enumerate(coeffs_matrix)]
    equation = ''.join(truth)[:-3]
    seq_parts15 = sequence_to_parts(seq, 15)
    seq_parts25 = sequence_to_parts(seq, 25)
    file_content15 = ""
    file_content25 = ""
    for seq_part in seq_parts15:
        file_content15 += f"[INST] Could you give me a linear equation for the following number sequence: {','.join([str(i) for i in seq_part])} [/INST] [RESP] Certainly, the equation is the following: {equation} [/RESP]\n"
    for seq_part in seq_parts25:
        file_content25 += f"[INST] Could you give me a linear equation for the following number sequence: {','.join([str(i) for i in seq_part])} [/INST] [RESP] Certainly, the equation is the following: {equation} [/RESP]\n"
    return file_content15, file_content25

example = "[INST] Could you give me a linear equation for the following number sequence: 0,0,1,2,4,7,12, ... [/INST] [RESP] Certainly, the equation is the following: a_n = 1*a_{n-1} + 1*a_{n-2} [/RESP]"


file1, file2 = column_to_prompts('A000051')
print(file1)



