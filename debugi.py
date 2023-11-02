import sympy as sp
import numpy as np
import pandas as pd
import exact_ed as ed

ed.check_eq_man(sp.Matrix([[1], [-1], [1]]), 'A000085', pd.read_csv('cores_test.csv'), solution_ref=['n*a(n-2)', 'a(n-2)', 'a(n-1)'], header=False)