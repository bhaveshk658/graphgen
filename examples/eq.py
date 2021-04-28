from math import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc

M = 1 # 0.5
s1 = 5 # 0.5
k = 0.005
N = 20
s2 = 5

s = s1**2 + s2**2 # 50
A = -(M * N) / sqrt(2*pi*s) # -0.4, -1.1283791670955126 (M=1,s1=5)

f_inv = (lambda y: y - ((A*y*exp((-(y**2)/(2*s)))) / (k*s)))
plt.figure(1)
y_arr = np.arange(0, 50, 0.1)
x_arr = [f_inv(y) for y in y_arr]
plt.plot(y_arr, x_arr)

# plt.figure(2)
# f = inversefunc(f_inv)
# x_arr = np.arange(0, 20, 0.1)
# y_arr = [f(x) for x in x_arr]
# print(y_arr)
# plt.plot(x_arr, y_arr)
plt.show()


