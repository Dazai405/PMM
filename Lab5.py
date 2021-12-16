import numpy as np
import time
from math import sqrt, cos, sin , log ,fabs , tan ,exp
import matplotlib.pyplot as plt
from numpy import log as ln
from mpl_toolkits.mplot3d import Axes3D

def lambda1(x):
    return 5500 / (560 + x) + 0.942 * (1e-10) * x * sqrt(x)

time_end= 2
L = 1
h=0.02
N =int(L/h)
D =1
T_1 = 0
T_n = 0
T_0 = 0
T1 =  np.zeros(N)
for i in range(0,N):
    T1[i] = T_0
    if i == 19: T1[i] = 100
T2 =  np.zeros(N)
T1[0] = T_1
T1[N-1] = T_n
x =  np.zeros(N)
tao = (h*h)/(D*4)
q = ((tao * D)/(h*h))
xx= 0
i=0
while(xx<=L):
    x[i] = xx
    i=i+1
    xx=xx+h
time = 0
plt.ion()
while(time<time_end):
    MAX = 0
    for i in range(0,N): T2[i] = T1[i]
    for i in range(0, N):
        if (T1[i] >= MAX): MAX =T1[i]
    tao = (h * h) / (lambda1(MAX) * 4)
    for i in range(1,N-1):
        T1[i] = (tao/h)*((T2[i+1] -T2[i])*lambda1(T2[i]) - (T2[i] -T2[i-1])*lambda1(T2[i])) + T2[i]
    print(tao)
    time = time + tao
    T1[0] =  T1[1]
    T1[N-1] = T1[N-2]
    plt.clf()
    plt.plot(x, T1)
    plt.draw()
    plt.gcf().canvas.flush_events()
plt.ioff()
plt.show()
