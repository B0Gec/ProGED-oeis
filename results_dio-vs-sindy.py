diolist = [
    " A000032 .. h .. a(n) = a(n - 2) + a(n - 1)",
    " A000035 .. h .. a(n) = a(n - 2)",
    " A000045 .. h .. a(n) = a(n - 2) + a(n - 1)",
    " A000079 .. h .. a(n) = 2*a(n - 1)",
    " A000129 .. h .. a(n) = a(n - 2) + 2*a(n - 1)",
    " A000204 .. h .. a(n) = a(n - 2) + a(n - 1)",
    " A000225 .. h .. a(n) = -2*a(n - 2) + 3*a(n - 1)",
    " A000244 .. h .. a(n) = 3*a(n - 1)",
    " A000290 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A000292 .. h .. a(n) = -a(n - 15) + a(n - 14) + a(n - 12) - a(n - 10) - a(n - 5) + a(n - 3) + a(n - 1)",
    " A000326 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A000330 .. h .. a(n) = -a(n - 4) + 4*a(n - 3) - 6*a(n - 2) + 4*a(n - 1)",
    " A000578 .. h .. a(n) = -a(n - 5) + 3*a(n - 4) - 2*a(n - 3) - 2*a(n - 2) + 3*a(n - 1)",
    " A000583 .. h .. a(n) = a(n - 5) - 5*a(n - 4) + 10*a(n - 3) - 10*a(n - 2) + 5*a(n - 1)",
    " A001045 .. h .. a(n) = 2*a(n - 2) + a(n - 1)",
    " A001057 .. h .. a(n) = a(n - 3) + a(n - 2) - a(n - 1)",
    " A001333 .. h .. a(n) = a(n - 2) + 2*a(n - 1)",
    " A001519 .. h .. a(n) = -a(n - 2) + 3*a(n - 1)",
    " A001906 .. h .. a(n) = -a(n - 2) + 3*a(n - 1)",
    " A002275 .. h .. a(n) = -10*a(n - 2) + 11*a(n - 1)",
    " A002378 .. h .. a(n) = a(n - 6) - a(n - 5) - a(n - 4) + a(n - 2) + a(n - 1)",
    " A002530 .. h .. a(n) = -a(n - 4) + 4*a(n - 2)",
    " A002531 .. h .. a(n) = -a(n - 4) + 4*a(n - 2)",
    " A002620 .. h .. a(n) = a(n - 8) - a(n - 7) - a(n - 5) + a(n - 3) + a(n - 1)",
    " A004526 .. h .. a(n) = -a(n - 3) + a(n - 2) + a(n - 1)",
    " A005408 .. h .. a(n) = -a(n - 2) + 2*a(n - 1)",
    " A005843 .. h .. a(n) = -a(n - 3) + a(n - 2) + a(n - 1)",
]

sindilist = [
    " A000032 .. h .. a(n) = a(n - 2) + a(n - 1)",
    " A000035 .. h .. a(n) = a(n - 2)",
    " A000045 .. h .. a(n) = a(n - 2) + a(n - 1)",
    " A000079 .. h .. a(n) = 2*a(n - 1)",
    " A000124 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A000129 .. h .. a(n) = a(n - 2) + 2*a(n - 1)",
    " A000204 .. h .. a(n) = a(n - 2) + a(n - 1)",
    " A000217 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A000225 .. h .. a(n) = -2*a(n - 2) + 3*a(n - 1)",
    " A000244 .. h .. a(n) = 3*a(n - 1)",
    " A000290 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A000302 .. h .. a(n) = 4*a(n - 1)",
    " A000326 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A000330 .. h .. a(n) = -a(n - 4) + 4*a(n - 3) - 6*a(n - 2) + 4*a(n - 1)",
    " A000578 .. h .. a(n) = -a(n - 4) + 4*a(n - 3) - 6*a(n - 2) + 4*a(n - 1)",
    " A000583 .. h .. a(n) = a(n - 5) - 5*a(n - 4) + 10*a(n - 3) - 10*a(n - 2) + 5*a(n - 1)",
    " A001045 .. h .. a(n) = 2*a(n - 2) + a(n - 1)",
    " A001057 .. h .. a(n) = a(n - 3) + a(n - 2) - a(n - 1)",
    " A001333 .. h .. a(n) = a(n - 2) + 2*a(n - 1)",
    " A001519 .. h .. a(n) = -a(n - 2) + 3*a(n - 1)",
    " A001906 .. h .. a(n) = -a(n - 2) + 3*a(n - 1)",
    " A002275 .. h .. a(n) = -10*a(n - 2) + 11*a(n - 1)",
    " A002378 .. h .. a(n) = a(n - 3) - 3*a(n - 2) + 3*a(n - 1)",
    " A002530 .. h .. a(n) = -a(n - 4) + 4*a(n - 2)",
    " A002531 .. h .. a(n) = -a(n - 4) + 4*a(n - 2)",
    " A002620 .. h .. a(n) = a(n - 4) - 2*a(n - 3) + 2*a(n - 1)",
    " A004526 .. h .. a(n) = -a(n - 4) + 2*a(n - 2)",
    " A005408 .. h .. a(n) = -a(n - 2) + 2*a(n - 1)",
    " A005843 .. h .. a(n) = -a(n - 2) + 2*a(n - 1)",
]

print(diolist)
same = [i for i in diolist if i in sindilist]
diff = [('dio', i) for i in diolist if i not in sindilist] + [('sindy', i) for i in sindilist if i not in diolist]
print(len(diolist), 'diolist')
print(len(sindilist), 'sindilist')
print(len(same), same)
print(len(diff), diff)
print('sindilist')
for i in sindilist:
    print(i)
print('diolist')
for i in diolist:
    print(i)
for i in diff:
    print(i)




