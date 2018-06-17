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

def MINE(Start: np.ndarray):
    Cycle = 100
    Digits = 2
    a, b = 2, 1
    MemNum = 30
    x0 = mat.repmat(Start, MemNum, 1)
    Dir = a * np.random.rand(MemNum, Digits) - b
    x0 += Dir 
    Dir = Dir / Dir.min()
    if Dir.dtype != "float64":
        Dir = Dir.astype("float")
    Dir = Dir / (la.norm(Dir, axis=1).reshape(MemNum, 1))
    Hbest = np.zeros((MemNum, Digits))
    Hbestv = np.zeros((MemNum, 1))
    for inner in range(MemNum):
        Hbestv[inner,:] = Evaluate(x0[inner,:])
        Hbest[inner,:] = x0[inner,:]
    T0, Tend = 0.45, 0.9
    A = (Tend - T0) / (Cycle ** 0.5)
    B = T0
    H = []
    Pace = np.std(Hbest, axis=0)
    Pace = 3 * la.norm(Pace)
    # P = []
    # R = []
    # Po = []
    for i in range(Cycle):
        radius = np.std(Hbest, axis=0)
        po = np.mean(Hbest,axis=0)
        # Po.append(po)
        radius = la.norm(radius)
        T = B + A * (i ** 0.5)
        # Pace = 40 / (1 + np.exp(7 * (i - (4*Cycle /3 )) / Cycle))
        Judge = 1 / (1 + np.exp(Pace /(radius+1e-15) - 1))
        Pace = 3.5 * radius * Judge
        Bias = 1.2 * Judge
        # P.append(Pace)
        # R.append(radius)
        # Bias = 1.2 / (1 + np.exp(-(i - (Cycle / 3)) / Cycle))
        Sort = np.argsort(Hbestv, axis=0)
        Chaos = 0.4 * np.random.random(Digits) + 0.8
        Chaos.reshape(Digits, 1)
        Transfer = np.zeros((MemNum, MemNum))
        for Index in range(MemNum):
            Transfer[Index, Sort[Index, :]] = 1
        Ref = np.dot(Transfer, Hbest)
        for Inner in range(MemNum):
            base = Ref - mat.repmat(x0[Inner,:], MemNum, 1)
            Normal = la.norm(base)
            det = Normal.argmin()
            base = np.delete(base, det, axis=0)
            base = base[0:Digits,:]
            if base.dtype != "float":
                base = base.astype("float")
            norm = la.norm(base,axis=1).reshape(Digits,1)
            for index in range(Digits):
                if norm[index, 0] == 0:
                    base[index,:] = np.ones(Digits)
                else:
                    base[index,:] = base[index,:] / norm[index,0]
            Hbelief = 2 * np.random.rand(1, Digits) - Bias * Chaos
            Direction = np.dot(Hbelief, base)
            Direction = Direction / la.norm(Direction)
            Gbelief = np.random.logistic(T,(1-T)/3, Digits)
            Dir[Inner,:] =np.abs(1 - Gbelief)  * Dir[Inner,:] + Gbelief * Direction
            if Dir[Inner,:].dtype != "float64":
                Dir = Dir.astype("float")
            Dir[Inner,:] = Dir[Inner,:] / la.norm(Dir[Inner,:])
            x0[Inner,:] = x0[Inner,:] + Pace * Dir[Inner,:]
            Check = Evaluate(x0[Inner,:])
            if Hbestv[Inner,:] > Check:
                Hbest[Inner,:] = x0[Inner,:]
                Hbestv[Inner,:] = Check
        yield Hbest,po,radius


    

fig = plt.figure(figsize=(200,200))
# ax = plt.axes(xlim=(-1000,1200), ylim=(-1000, 1200))
ax = plt.axes(xlim=(-1000,1200), ylim=(-1000, 1200))
line, =plt.plot([], [], ".")
line2, = plt.plot([], [], linewidth=0.5,c="r")
plt.grid()
def init():
    line.set_data([],[])
    line2.set_data([],[])
    return line,line2

def animate(X):
    x, y = X[0][:,0],X[0][:,1]
    line.set_data(x, y)
    Theta = np.linspace(0, 2 * np.pi, 1000)
    Xc, Yc = X[1][0] + X[2] * np.cos(Theta), X[1][1] + X[2] * np.sin(Theta)
    line2.set_data(Xc,Yc)
    return line,line2

animMINE = animation.FuncAnimation(fig, animate,MINE(1000*np.ones((1,Digits))), interval=200,repeat=True,init_func=init)
plt.show()