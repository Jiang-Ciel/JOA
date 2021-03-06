from numpy import linalg as la
from numpy import matlib as mat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
Digits = 200
MemNum = 300
Cycle = 250
font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 14,
        }
def Evaluate(x):
    i = np.arange(Digits) + 1
    y = np.cos(x / (i ** 0.5))
    return 1 + la.norm(x)**2/4000 - y.prod()

def Get_Direction(Ref, x0, Hbelief):
    base = Ref - np.array(x0)
    Normal = la.norm(base, axis=1).reshape(MemNum,1)
    det = Normal.argmin()
    base = np.delete(base, det, axis=0)
    Normal = np.delete(Normal, det, axis=0)
    base = base[0:Digits,:]
    Normal = Normal[0:Digits,:]
    if base.dtype != "float":
        base = base.astype("float")
    base = base / Normal
    return np.dot(np.array(Hbelief), base).reshape(Digits,)

def MINE(Start: np.ndarray):
    a, b = 20, 10
    alpha = 1.5
    # MemNum = 0
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
    H = []
    radius0 = np.std(Hbest, axis=0)
    radius0 = la.norm(radius0)
    Pace = 3 * radius0
    P = []
    R = []
    Po = []
    Ratio = []
    Bi = []
    Direction = np.zeros((MemNum, Digits))
    APace = alpha * (1 + np.exp(alpha - 1))
    for i in range(Cycle):
        radius1 = np.std(Hbest, axis=0)
        po = np.mean(Hbest,axis=0)
        Po.append(po)
        radius1 = la.norm(radius1)
        Judge = 1 / (1 + np.exp(Pace / (radius1 + 1e-15) - 1))
        Trust = 1 / (1 + np.exp(alpha * Judge - 1))
        print("ratio:",radius1/radius0)
        print("Trust:",Trust)
        print("Judge:",Judge)
        Pace = APace * radius1 * Judge
        Pace = alpha * (1 + np.exp(alpha - 1)) * radius1 * Judge
        Bias = 1.6 * Judge
        radius0 = radius1
        Ratio.append(Pace / radius0)
        Bi.append(Bias)
        P.append(Pace)
        R.append(radius0)
        Sort = np.argsort(Hbestv, axis=0)
        Chaos = 0.4 * np.random.random(Digits) + 0.8
        Chaos.reshape(1,Digits)
        Ref = Hbest[Sort[:,0],:]
        Hbelief = (2 * np.random.rand(MemNum, Digits) - Bias * Chaos).tolist()
        Direction = np.array(list(map(Get_Direction, [Ref for _ in range(MemNum)], x0.tolist(), Hbelief)))
        Direction = Direction / (la.norm(Direction,axis=1).reshape(MemNum,1))
        Gbelief = np.random.normal(Trust,min([1-Trust,Trust])/3, (MemNum,Digits))
        Dir =np.abs(1 - Gbelief)  * Dir + Gbelief * Direction
        if Dir.dtype != "float64":
            Dir = Dir.astype("float")
        Dir = Dir / (la.norm(Dir,axis=1).reshape(MemNum,1))
        x0 = x0 + Pace * Dir
        Check = np.array(list(map(Evaluate, x0))).reshape(MemNum, 1)
        Place = np.where(Check - Hbestv < 0)
        Hbestv[Place,:] = Check[Place,:]
        Hbest[Place,:] = x0[Place,:]
        h = Hbest[np.argmin(Hbestv),:].tolist()
        print("MINE", i + 1)
        print("--------------------")
        print("Obj.Func:", np.min(Hbestv))
        print("Componets MAX:", np.max(np.abs(Hbest[np.argmin(Hbestv),:])))
        print("Componets MIN:", np.min(np.abs(Hbest[np.argmin(Hbestv),:])))
        print('====================')
        h.append(Hbestv.min())
        H.append(h)
    Po = np.array(Po)
    plt.figure("MINE：Time-Position")
    for i in range(Digits):
        plt.plot(np.arange(Cycle),Po[:,i])
    plt.xlabel("Times")  
    plt.ylabel("Components") 
    plt.figure("MINE：Time-Radius/Pace")
    Radius,=plt.plot(range(Cycle), R, label="Radius")
    Pace,=plt.plot(range(Cycle), P,linestyle=":", label="Pace")
    plt.legend(loc="upper right")
    plt.xlabel("Times")  
    plt.ylabel("Value")
    plt.figure("MINE：Time-Radius/Pace,Bias")
    ratio,=plt.plot(range(Cycle), Ratio, linestyle=":",label="Pace/Radius")
    Bias,=plt.plot(range(Cycle), Bi, label="Bias")
    plt.legend(loc="upper right")
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of 'Pace/Radius' and 'Bias'",fontdict=font)
    H = np.array(H)
    value = H[:,Digits]
    value.reshape(1,Cycle)
    return value,Hbestv.min(),Hbest[np.argmin(Hbestv),:]

def PSO(Start:np.ndarray):
    GroupNum = 300
    Coefficient = mat.repmat(Start,GroupNum,1) + 20* np.random.rand(GroupNum, Digits) - 10
    v0 = 40 * np.random.rand(GroupNum, Digits) - 20
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
    plt.xlabel("Times")  
    plt.ylabel("Components") 
    return value,pbestv.min(),pbest[np.argmin(pbestv),:]

VMINE,vMINE,LocMINE = MINE(500*np.ones((1,Digits)))
VPSO, vPSO, LocPSO = PSO(500 * np.ones((1, Digits)))
plt.figure("Time-F(x)")
MINE, =plt.plot(range(Cycle), VMINE,color='red',label="MINE")
PSO, =plt.plot(range(Cycle), VPSO,linestyle=":", color="blue", label="PSO")
print("Function:MINE", "Evaluation:", vMINE)
print("Location:", LocMINE)
print("===========")
plt.figure("Components Final")
M = plt.scatter(range(Digits), LocMINE,label="MINE")
P = plt.scatter(range(Digits), LocPSO, label="PSO")
plt.legend(loc='upper right')
plt.xlabel("digits")  
plt.ylabel("Components")==========")
print("Function:PSO ", "Evaluation:", vPSO)
print("Location:", LocPSO)
plt.legend(loc='upper right')
plt.xlabel("times")  
plt.ylabel("f(x)")

plt.show()