import time
import numpy
import math

scale = 8

start = time.perf_counter()
for i in range(10**scale):
    math.sin(i)
print(time.perf_counter() - start)

start = time.perf_counter()
for i in range(10**scale):
    numpy.sin(i)
print(time.perf_counter() - start)
