def a079034(n):
    return (n ** 4 - n ** 2 + 12) / 12

sepa = 140
print(str(list(map(a079034, [i for i in range(120)])))[:sepa])
print(str(list(map(a079034, [i for i in range(120)])))[sepa:2*sepa])
print(str(list(map(a079034, [i for i in range(120)])))[2*sepa:])
