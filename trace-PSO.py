from numpy import linalg as la
from numpy import matlib as mat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
Digits = 2
Cycle = 100
def Evaluate(estimate):
    estimate = estimate.reshape(1, Digits)
    A = la.norm(estimate)** 2
    B = 10 * np.cos(estimate * 2 * np.pi)
    return 10 * Digits + A - B.sum()

def PSO(Start:np.ndarray):
    GroupNum = 30
    digits = 2
    x0 = mat.repmat(Start,GroupNum,1) + 20* np.random.rand(GroupNum, digits) - 10
    v0 = 40 * np.random.rand(GroupNum, digits) - 20
    Cycle = 100
    pbest = x0.view()
    pbestv = np.zeros((GroupNum, 1))
    pbestv1 = np.zeros((GroupNum, 1))
    for Inner in range(GroupNum):
        Check = Evaluate(x0[Inner,:])
        pbestv[Inner,:] = Check
    gbest = mat.repmat(x0[np.argmin(pbestv),:], GroupNum, 1)
    wmax, wmin, c1, c2 = 0.9, 0.4, 2, 2
    for i in range(Cycle):
        w = wmax - i * (wmax - wmin) / Cycle
        r1 = np.random.rand(GroupNum, 1)
        r2 = np.random.rand(GroupNum, 1)
        v1 = w * v0 + c1 * r1 * (pbest - x0) + c2 * r2 * (gbest - x0)
        x0 = x0 + v1
        for Inner in range(GroupNum):
            Check = Evaluate(x0[Inner,:])
            pbestv1[Inner,:] = Check 
        for j in range(GroupNum):
            if pbestv1[j,:] < pbestv[j,:]:
                pbest[j,:] = x0[j,:]
                pbestv[j,:] = pbestv1[j,:]
        v0 = v1.view()
        gbest = mat.repmat(pbest[np.argmin(pbestv),:], GroupNum, 1)
        yield pbest 
    

fig = plt.figure(figsize=(200,200))
ax = plt.axes(xlim=(-1000,1200), ylim=(-1000, 1200))
line, =plt.plot([], [], ".")
plt.grid()
def init():
    line.set_data([],[])
    return line,

def animate(X):
    x, y = X[:,0],X[:,1]
    line.set_data(x,y)
    return line,

animPSO = animation.FuncAnimation(fig, animate,PSO(1000*np.ones((1,Digits))), interval=200,repeat=True,init_func=init)
# animMINE = animation.FuncAnimation(fig, animate,MINE(1000*np.ones((1,Digits))), interval=200,repeat=True,init_func=init)
plt.show()