import matplotlib.pyplot as pt
import numpy as np

# x = np.linspace(0, 10, 10000)
# y = np.log(x)

# pt.plot(x, y)
# pt.show()


x = np.linspace(0, 10, 10)
y = np.log(x)

pt.scatter(x, y)
pt.show()

xs = np.array([1])
for i in range(-3, 2):
    x = np.linspace(10**(i), 10**(i+1), 10)
    xs = np.hstack((xs, x))
print(xs)
y = np.log(xs)
print(xs)

pt.scatter(xs, y, s=1)
# print(np.zeros((0, y.shape[0])))
pt.scatter(xs, np.zeros((1, y.shape[0])).flatten(), s=1, c='black')
pt.scatter(np.zeros((1, y.shape[0])).flatten(), y, s=1)
pt.scatter(np.zeros((1, y.shape[0])).flatten(), y, s=1, c='red')
# pt.scatter(np.zeros((1, y.shape[0])).flatten(), y, s=1, c=(58, 58, 216))
pt.show()
print(2)
