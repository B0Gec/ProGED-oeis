"""
This module contains preprossing and postprocessing code to help with the implementation of
the functions to compute two linear recursive equations are equivalent.
"""

import re

from eq_ideal import linear_to_vec, is_linear

# def is_linear_w_const(expr: str) -> bool:
#     """Check if the expression is linear allowing also the constant term.
#         It is copy of first part of the is_linear code in eq_ideal which does not allow constants.
#     """
#
#     if '^' in expr:  # Power to > 1 indicates nonlinearity.
#         return False
#     else:  # Check if there are degree >= 2 terms.
#         quads = re.findall(r'[an][(n\-0-9)]*\*[an][(n\-0-9)]*', expr)  # for linear including 'n'
#         if quads:
#             return False
#
#     return True


# diofa: a(n) = -10*a(n - 4) + a(n - 3) + 10*a(n - 1)

file_content = """
A002278:
a(n) -14*a(n-1) +84*a(n-2) -280*a(n-3) +560*a(n-4) -672*a(n-5) +448*a(n-6) -128*a(n-7)
truth:
a(n) = 11*a(n - 1) + -10*a(n - 2),
a(0) = 0, a(1) = 4

False  -  checked against website ground truth.
True  -  "manual" check if equation is correct.
"""

# eq =
# a(n) -14*a(n-1) +84*a(n-2) -280*a(n-3) +560*a(n-4) -672*a(n-5) +448*a(n-6) -128*a(n-7)

