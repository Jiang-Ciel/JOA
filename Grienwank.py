from numpy import linalg as la
from numpy import matlib as mat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
Digits = 15
def Evaluate(x):
    i = np.arange(Digits) + 1
    y = np.cos(x / (i ** 0.5))
    return 1 + la.norm(x)**2/4000 - y.prod()

Cycle = 100
font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 14,
        }

def PSO(Start:np.ndarray):
    GroupNum = 50
    # Digits = 15
    Coefficient = mat.repmat(Start, GroupNum, 1)+ 20 * np.random.rand(GroupNum, Digits) - 10
    v0 = 20 * np.random.rand(GroupNum, Digits) - 10
    Cycle = 100
    pbest = Coefficient.view()
    pbestv = np.zeros((GroupNum, 1))
    pbestv1 = np.zeros((GroupNum, 1))
    for Inner in range(GroupNum):
        Check = Evaluate(Coefficient[Inner,:])
        pbestv[Inner,:] = Check
    gbest = mat.repmat(Coefficient[np.argmin(pbestv),:], GroupNum, 1)
    wmax, wmin, c1, c2 = 0.9, 0.4, 2, 2
    value = []
    Po = []
    R = []
    P = []
    for i in range(Cycle):
        w = wmax - i * (wmax - wmin) / Cycle
        r1 = np.random.rand(GroupNum, 1)
        r2 = np.random.rand(GroupNum, 1)
        v1 = w * v0 + c1 * r1 * (pbest - Coefficient) + c2 * r2 * (gbest - Coefficient)
        pace = la.norm(v1,axis=1)
        pace = np.mean(pace)
        P.append(pace)
        Coefficient = Coefficient + v1
        radius = np.std(pbest, axis=0)
        radius = la.norm(radius)
        R.append(radius)
        po = np.mean(pbest, axis=0)
        Po.append(po)
        Coefficient = Coefficient +  v1
        for Inner in range(GroupNum):
            Check = Evaluate(Coefficient[Inner,:])
            pbestv1[Inner,:] = Check 
            if Check < pbestv[Inner,:]:
                pbest[Inner,:] = Coefficient[Inner,:]
                pbestv[Inner,:] = Check
        v0 = v1.view()
        gbest = mat.repmat(pbest[np.argmin(pbestv),:], GroupNum, 1)
        value.append(np.min(pbestv))
    Po = np.array(Po)
    R = np.array(R)
    P = np.array(P)
    plt.figure("PSO：Time-Position")
    for i in range(Digits):
        plt.plot(np.arange(Cycle), Po[:, i])
    plt.xlabel("Step",fontdict=font)  
    plt.ylabel("Value of object function", fontdict=font) 
    # plt.figure("PSO：Time-Radius/Pace")
    # Pace,=plt.plot(range(Cycle), P, label="Pace")
    # Radius,=plt.plot(range(Cycle), R, label="Radius")
    # plt.legend(loc="upper right")
    # plt.xlabel("Times")  
    # plt.ylabel("Value") 
    return value,pbestv.min(),pbest[np.argmin(pbestv),:]

def MINE(Start: np.ndarray):
    Cycle = 100
    # Digits = 15
    a, b = 20, 10
    MemNum = 50
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
    value = []
    Pace = np.std(Hbest, axis=0)
    Pace = 4 * la.norm(Pace)
    P = []
    R = []
    Po = []
    Ratio = []
    Bi = []
    for i in range(Cycle):
        po = np.mean(Hbest, axis=0)
        Po.append(po)
        radius = np.std(Hbest, axis=0)
        radius = la.norm(radius)
        Ratio.append(Pace/radius)
        Judge = 1 / (1 + np.exp(Pace /(radius+1e-15) - 1))
        Pace = 4 * radius *Judge
        Bias = 1.2 * Judge
        Bi.append(Bias)
        P.append(Pace)
        R.append(radius)
        # Bias = 1.2 / (1 + np.exp(2 * (i - (Cycle / 2)) / Cycle))
        Sort = np.argsort(Hbestv, axis=0)
        c1 = np.linspace(1.2, 1, Digits)
        c1.reshape(Digits, 1)
        T0, Tend = 0.45,0.9
        A = (Tend - T0) / (Cycle ** 0.5)
        B = T0
        Ref = Hbest[Sort[:, 0],:]
        for Inner in range(MemNum):
            T = B + A * (Inner ** 0.5)
            base = Ref - mat.repmat(x0[Inner,:], MemNum, 1)
            Normal = la.norm(base, axis=0)
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
            belief1 = 2 * np.random.rand(1, Digits) - Bias * c1
            belief2 = np.random.logistic(T,np.abs((1-T)/3), Digits)
            direction = np.dot(belief1, base)
            direction = direction / la.norm(direction)
            Dir[Inner,:] = (1-belief2) * Dir[Inner,:] + belief2 * direction
            if Dir[Inner,:].dtype != "float64":
                Dir = Dir.astype("float")
            Dir[Inner,:] = Dir[Inner,:] / la.norm(Dir[Inner,:])
            x0[Inner,:] = x0[Inner,:] + Pace * Dir[Inner,:]
            Check = Evaluate(x0[Inner,:])
            if Hbestv[Inner,:] > Check:
                Hbest[Inner,:] = x0[Inner,:]
                Hbestv[Inner,:] = Check
        value.append(Hbestv.min())
    Po = np.array(Po)
    plt.figure("MINE：Time-Position")
    for i in range(Digits):
        plt.plot(np.arange(Cycle), Po[:, i])
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of every Digits",fontdict=font)
    plt.figure("MINE：Time-Radius/Pace")
    Radius,=plt.plot(range(Cycle), R, linestyle=":", label="Radius")
    Pace,=plt.plot(range(Cycle), P, label="Pace")
    plt.legend(loc="upper right")
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of 'Pace' and 'Radius'", fontdict=font)
    plt.figure("MINE：Time-Radius/Pace,Bias")
    ratio,=plt.plot(range(Cycle), Ratio, linestyle=":",label="Pace/Radius")
    Bias,=plt.plot(range(Cycle), Bi, label="Bias")
    plt.legend(loc="upper right")
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of 'Pace/Radius' and 'Bias'",fontdict=font)
    return value,Hbestv.min(),Hbest[np.argmin(Hbestv),:]

VMINE,vMINE,LocMINE = MINE(1000 * np.ones((1, Digits)))
VPSO, vPSO, LocPSO = PSO(1000 * np.ones((1, Digits)))
plt.figure("Time-F(x)")
MINE, =plt.plot(range(Cycle), VMINE,color='red',label="MINE")
PSO, =plt.plot(range(100), VPSO,linestyle=":",color="blue",label="PSO")
print("Function:MINE", "Evaluation:", vMINE)
print("Location:", LocMINE)
print("=====================")
print("Function:PSO ", "Evaluation:", vPSO)
print("Location:", LocPSO)
plt.legend(loc='upper right')
plt.xlabel("Step",fontdict=font)  
plt.ylabel("Value of object function", fontdict=font)  
plt.show()

