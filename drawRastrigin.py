from mpl_toolkits.mplot3d import Axes3D  
import numpy as np  
from matplotlib import pyplot as plt  
  
fig = plt.figure()  
ax = Axes3D(fig)  
x=np.arange(-10,10,0.2)  
y=np.arange(-10,10,0.2) 
X, Y = np.meshgrid(x, y)#网格的创建，这个是关键  
Z = 20+X**2+Y**2-10*(np.cos(2*np.pi*X)+np.cos(2*np.pi*Y))
# Z = X**2+Y**2
# Z = 1 + (X**2+Y**2)/4000 - np.cos(X)*np.cos(Y/(2**0.5))
# Z= 10*X*(Y**5)
# Z = np.abs(20-10*(np.cos(2*np.pi*X)+np.cos(2*np.pi*Y))+np.sin(0.0001*(X**2+Y**2)**0.5))*np.sin(0.000001*(X**2+Y**2)**0.5)
# Z = np.sin((X**2+Y**2)*np.pi/10020)*(0.5 + (np.sin(np.pi*(X ** 2 - Y ** 2)) - 0.5) / ((1 + 0.001 * (X ** 2 + Y ** 2))** 2))
# Z = 0.5-(np.sin(3*(X**2-Y**2))**2-0.5)/((1+0.001*(X**2+Y**2))**2)
# a = la.norm(estimate)
# Z = (((X/20)**2+(Y/20)**2) / (((X/20)**2+(Y/20)**2)**2+1))*(np.abs(np.sin(2*np.pi*((X/20)**2+(Y/20)**2)**0.5))+0.5)
plt.xlabel('x')  
plt.ylabel('y')
ax.set_zlabel('f(x,y)') 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')  
plt.show()