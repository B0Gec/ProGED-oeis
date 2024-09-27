"""
Copied from test_grobne.py
Testing Moeller-Buchberger's algorithm in CoCoa, by printing code here and then run it in CoCoa.
Look testmb.cocoa5

Tested:
- fibonaci: NOT working! (not sure why)
- y = x + 3: working
- y = 2x : working?
- y = x + 3 in 3D (x, y, z) space: working
- y = x**2 : NOT working!
"""


# trivial:

# y = x^2, i.e. p(x,y) = y- x^2 = 0
print(f"Points := mat({[[i, i**2] for i in range(10)]});")
# success!! :
# ideal(x^2 -y,  x*y^4 +(-1/45)*y^5 +210*x*y^3 +(-58/3)*y^4 +5985*x*y^2 +(-21091/15)*y^3 +26060*x*y +(-144736/9)*y^2 +8064*x +(-114064/5)*y,  y^6 -495*y^5 -3659040*x*y^3 +212223*y^4 -146361600*x*y^2 +30364235*y^3 -728148960*x*y +425883876*y^2 -239500800*x +661210560*y)

# fibonaci:
print(f"Points := mat({[[i, i**2] for i in range(10)]});")
#     [
# [0,  1,  1],
# [1,  1,  2],
# [1,  2,  3],
# [2,  3,  5],
# [3,  5,  8],
# ]
#
# success!! a_{n+2} = a_{n+1} + a_n
# ideal(x +y -z,  y^2 +(-7/6)*y*z +(1/3)*z^2 +(-1/6)*y +(1/6)*z -1/6,  z^3 -36*y*z +9*z^2 +78*y +2*z -54,  y*z^2 -28*y*z +9*z^2 +55*y -2*z -35)


# line y= 2x:
y2x = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32]]
# succsess!!:
# ideal(x +(-1/2)*y,  y^5 -62*y^4 +1240*y^3 -9920*y^2 +31744*y -32768)


# line y= x + 3: (finds)
# [ [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], ]

# line y= x + 3: (finds) in more variables:
# [0, 3,  0], [3, 6,  0], [6, 9,  0], [9, 12, 0], [12, 15, 0],

# line y= x**2:
# [0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81], [10, 100]


# Points := mat([[- 1, - 0], [- 0, - 1], [+ 1, + 0], [- 0, + 1], [- 1/2, sp.sqrt(3)/2]]);

# all the cases of testing mavi (including failures) were by BM algorithm 100% successfull!!!




