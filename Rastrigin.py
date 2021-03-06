from numpy import linalg as la
from numpy import matlib as mat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
Digits = 30
Cycle = 100
font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 14,
        }
def Evaluate(estimate):
    estimate = estimate.reshape(1, Digits)
    A = la.norm(estimate)** 2
    B = 10 * np.cos(estimate * 2 * np.pi)
    return 10 * Digits + A - B.sum()
  
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
    T0, Tend = 0.45, 0.7
    A = (Tend - T0) / (Cycle ** 0.5)
    B = T0
    H = []
    Pace = np.std(Hbest, axis=0)
    Pace = 3 * la.norm(Pace)
    P = []
    R = []
    Po = []
    Ratio = []
    Bi = []
    for i in range(Cycle):
        radius = np.std(Hbest, axis=0)
        po = np.mean(Hbest,axis=0)
        Po.append(po)
        radius = la.norm(radius)
        T = B + A * (i ** 0.5)
        # Pace = 40 / (1 + np.exp(7 * (i - (4*Cycle /3 )) / Cycle))
        Ratio.append(Pace/radius)
        Judge = 1 / (1 + np.exp(Pace /(radius+1e-15) - 1))
        Pace = 4 * radius *Judge
        Bias = 1.6 * Judge
        Bi.append(Bias)
        P.append(Pace)
        R.append(radius)
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
        h = Hbest[np.argmin(Hbestv),:].tolist()
        h.append(Hbestv.min())
        H.append(h)
    plt.figure("MINE：Time-Radius/Pace")
    Radius,=plt.plot(range(Cycle), R, linestyle=":",label="Radius")
    Pace,=plt.plot(range(Cycle), P, label="Pace")
    plt.legend(loc="upper right")
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of 'Pace' and 'Radius'",fontdict=font)
    plt.figure("MINE：Time-Radius/Pace,Bias")
    ratio,=plt.plot(range(Cycle), Ratio, linestyle=":",label="Pace/Radius")
    Bias,=plt.plot(range(Cycle), Bi, label="Bias")
    plt.legend(loc="upper right")
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of 'Pace/Radius' and 'Bias'",fontdict=font)
    Po = np.array(Po)
    plt.figure("MINE：Time-Position")
    for i in range(Digits):
        plt.plot(np.arange(Cycle), Po[:, i])
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of every Digits", fontdict=font)
    H = np.array(H)
    value = H[:,Digits]
    value.reshape(1,Cycle)
    return value,Hbestv.min(),Hbest[np.argmin(Hbestv),:]

def PSO(Start:np.ndarray):
    GroupNum = 50
    # Digits = 15
    Coefficient = mat.repmat(Start,GroupNum,1) + 20* np.random.rand(GroupNum, Digits) - 10
    v0 = 40 * np.random.rand(GroupNum, Digits) - 20
    Cycle = 100
    pbest = Coefficient.view()
    pbestv = np.zeros((GroupNum, 1))
    pbestv1 = np.zeros((GroupNum, 1))
    H = []
    for Inner in range(GroupNum):
        Check = Evaluate(Coefficient[Inner,:])
        pbestv[Inner,:] = Check
    gbest = mat.repmat(Coefficient[np.argmin(pbestv),:], GroupNum, 1)
    wmax, wmin, c1, c2 = 0.9, 0.4, 2, 2
    R = []
    Po = []
    P = []
    for i in range(Cycle):
        w = wmax - i * (wmax - wmin) / Cycle
        r1 = np.random.rand(GroupNum, 1)
        r2 = np.random.rand(GroupNum, 1)
        v1 = w * v0 + c1 * r1 * (pbest - Coefficient) + c2 * r2 * (gbest - Coefficient)
        Coefficient = Coefficient + v1
        pace = la.norm(v1,axis=1)
        pace = np.max(pace)
        P.append(pace)
        Coefficient = Coefficient + v1
        radius = np.std(pbest, axis=0)
        radius = la.norm(radius)
        R.append(radius)
        po = np.mean(pbest, axis=0)
        Po.append(po)
        for Inner in range(GroupNum):
            Check = Evaluate(Coefficient[Inner,:])
            pbestv1[Inner,:] = Check 
        for j in range(GroupNum):
            if pbestv1[j,:] < pbestv[j,:]:
                pbest[j,:] = Coefficient[j,:]
                pbestv[j,:] = pbestv1[j,:]
        v0 = v1.view()
        gbest = mat.repmat(pbest[np.argmin(pbestv),:], GroupNum, 1)
        h = pbest[np.argmin(pbestv),:].tolist()
        h.append(pbestv.min())
        H.append(h)
    H = np.array(H)
    out = pd.DataFrame(H)
    out.to_excel("Record.xlsx",sheet_name="PSO")
    value = H[:, Digits]
    value.reshape(1, Cycle)
    Po = np.array(Po)
    R = np.array(R)
    plt.figure("PSO：Time-Position")
    for i in range(Digits):
        plt.plot(np.arange(Cycle), Po[:, i])
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of every Digits",fontdict=font) 
    # plt.figure("PSO：Time-Radius/Pace")
    # Pace,=plt.plot(range(Cycle), P, label="Pace")
    # Radius,=plt.plot(range(Cycle), R, label="Radius")
    # plt.legend(loc="upper right")
    # plt.xlabel("Times")  
    # plt.ylabel("Value")
    return value,pbestv.min(),pbest[np.argmin(pbestv),:]

VMINE,vMINE,LocMINE = MINE(1000*np.ones((1,Digits)))
VPSO, vPSO, LocPSO = PSO(1000 * np.ones((1, Digits)))
plt.figure("Time-F(x)")
MINE, =plt.plot(range(Cycle), VMINE,color='red',label="MINE")
PSO, =plt.plot(range(Cycle), VPSO,linestyle=":", color="blue", label="PSO")
print("Function:MINE", "Evaluation:", vMINE)
print("Location:", LocMINE)
print("=====================")
print("Function:PSO ", "Evaluation:", vPSO)
print("Location:", LocPSO)
plt.legend(loc='upper right')
plt.xlabel("Step",fontdict=font)  
plt.ylabel("Value of object function", fontdict=font)
plt.figure("Components Final")
M = plt.scatter(range(Digits), LocMINE,label="MINE")
P = plt.scatter(range(Digits), LocPSO, label="PSO")
plt.legend(loc='upper right')
plt.xlabel("Digits")  
plt.ylabel("Components")
plt.show()