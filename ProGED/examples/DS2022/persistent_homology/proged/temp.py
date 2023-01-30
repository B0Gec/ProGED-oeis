
import os
import pickle

batch_path = "D:\\Experiments\\MLJ23\\proged\\structures\\v6\\batchsize10\\twostate1\\"
grammar_info_filename = "structures_v6_polynomial_twostate1_n10000_grammar_info.pkl"
with open(os.path.join(batch_path, grammar_info_filename), "rb") as file:
    structures = pickle.load(file)

