"Paper plane code"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Reading data of experimental results

M1L1 = pd.read_csv("M1L1.csv", sep="\s+", on_bad_lines='skip', header=None)
M1L2 = pd.read_csv("M1L2.csv", sep="\s+", on_bad_lines='skip', header=None)
M1L3 = pd.read_csv("M1L3.csv", sep="\s+", on_bad_lines='skip', header=None)
M2L1 = pd.read_csv("M2L1.csv", sep="\s+", on_bad_lines='skip', header=None)
M2L2 = pd.read_csv("M2L2.csv", sep="\s+", on_bad_lines='skip', header=None)
M2L3 = pd.read_csv("M2L3.csv", sep="\s+", on_bad_lines='skip', header=None)
M2L4 = pd.read_csv("M2L4.csv", sep="\s+", on_bad_lines='skip', header=None)
M2L5 = pd.read_csv("M2L5.csv", sep="\s+", on_bad_lines='skip', header=None)
M3L1 = pd.read_csv("M3L1.csv", sep="\s+", on_bad_lines='skip', header=None)
M3L2 = pd.read_csv("M3L2.csv", sep="\s+", on_bad_lines='skip', header=None)
M3L3 = pd.read_csv("M3L3.csv", sep="\s+", on_bad_lines='skip', header=None)

M3L1[2] = M3L1[2] - min(M3L1[2])

data = M3L1


data[1] = -data[1]

#Defined class paper plane to made the evolution of the system

class Paper_plane():

    def __init__(self, x0 = [0.0,0.0,0.0], v0 = [0.0,0.0,0.0], dt = 0.01, M = np.ones(9).reshape(3,3)/10):
        self.x = np.asarray(x0)
        self.v = np.asarray(v0)
        self.storex = [x0[0]]
        self.storey = [x0[1]]
        self.storez = [x0[2]]
        self.storevx = [v0[0]]
        self.storevy = [v0[1]]
        self.storevz = [v0[2]]
        self.dt = dt
        self.M = np.asarray(M)
        self.N = np.asarray(M)

    def MM(self, A, b):
        A = np.asarray(A, dtype= "float")
        b = np.asarray(b, dtype= "float")
        r = np.asarray([0,0,0], dtype= "float")
        for i in range(0,3):
            for j in range(0,3):
                r[i] = r[i] + A[i][j]*b[j]
        return r

    
    def Matrix(self, A = np.zeros(9).reshape(3,3), Ext = False):
        if Ext:
            self.M[0][2] = self.N[0][2]*self.v[0]
            self.M[2][0] = self.N[2][0]*self.v[2]
            return self.M
        else:
            self.M = self.M
        return self.M
    
    def force(self, x, v):
        beta = -0.2
        K_xx = -0.13
        K_xz = 0.1
        K_zx = 0.11
        K_zz = -0.02
        g = np.asarray([0.0, 0.0, -9.8])
        A = np.asarray(self.M)
        v = np.asarray(v)
        return  g + np.asarray([beta*v[0] + K_xx*v[0]**2 + K_xz*v[2]**2,0,
                                beta*v[2] + K_zx*v[0]**2 + K_zz*v[2]**2])

    
    def velocity(self, a, b):
        return b
    
    def RK4(self, a, b, p1, p2):

        l1 = a(p1,p2)
        k1 = b(p1,p2)

        l2 = a(p1 + self.dt*0.5*k1, p2 + self.dt*0.5*l1)
        k2 = b(p1 + self.dt*0.5*k1, p2 + self.dt*0.5*l1)

        l3 = a(p1 + self.dt*0.5*k2, p2 + self.dt*0.5*l2)
        k3 = b(p1 + self.dt*0.5*k2, p2 + self.dt*0.5*l2)

        l4 = a(p1 + self.dt*k3, p2 + self.dt*l3)
        k4 = b(p1 + self.dt*k3, p2 + self.dt*l3)

        return p1 + self.dt*(k1+2*k2+2*k3+k4)/6, p2 + self.dt*(l1+2*l2+2*l3+l4)/6
 

    def evol(self):
        if self.x[2] < 0:
            return
        else:
            self.x, self.v = self.RK4(self.force, self.velocity, self.x, self.v)
            self.storex.append(self.x[0])
            self.storey.append(self.x[1])
            self.storez.append(self.x[2])
            self.storevx.append(self.v[0])
            self.storevy.append(self.v[1])
            self.storevz.append(self.v[2])

    
    def data(self):
        return self.storex, self.storey, self.storez, self.storevx, self.storevy, self.storevz
    

#Border points interpolator

def vel(time, list):
    list = np.asarray(list)
    time = np.asarray(time)
    l = len(list)
    R = np.zeros(l)
    for i in range(1, l-1):
        R[i] = (list[i+1] - list[i-1])/(2*(time[i+1]- time[i]))
    R[0] = 3*R[1] - 3*R[2] + R[3]
    R[l-1] = 3*R[l-2] - 3*R[l-3] + R[l-4]
    return R

def acel(time, list):
    list = np.asarray(list)
    time = np.asarray(time)
    l = len(list)
    R = np.zeros(l)
    for i in range(2, l-2):
        R[i] = (-list[i+2] + 16*list[i+1] - 30*list[i] + 16*list[i-1] - list[i-2])/(12*(time[i+1]-time[i])**2)

    R[1] = 3*R[2] - 3*R[3] + R[4]
    R[l-2] = 3*R[l-3] - 3*R[l-4] + R[l-5]
    R[0] = 3*R[1] - 3*R[2] + R[3]
    R[l-1] = 3*R[l-2] - 3*R[l-3] + R[l-4]
    return R

data[3] = vel(data[0], data[1])
data[4] = vel(data[0], data[2])
data[5] = acel(data[0], data[1])
data[6] = acel(data[0], data[2])

vx_0 = data[3][2]
vz_0 = data[4][2]
ax_0 = data[5][2]
az_0 = data[6][2]

vx_1 = data[3][4]
vz_1 = data[4][4]
ax_1 = data[5][4]
az_1 = data[6][4]


#Calculating coeficient matrix


Axx = (ax_0 - ax_1*vz_0/vz_1)/(vx_0 - vx_1*vz_0/vz_1)
Axz = (ax_0 - ax_1*vx_1/vx_0)/(vz_0 - vz_1*vx_0/vx_1)
Azx = (az_0 - az_1*vz_0/vz_1)/(vx_0 - vx_1*vz_0/vz_1)
Azz = (az_0 - az_1*vx_1/vx_0)/(vz_0 - vz_1*vx_0/vx_1)

Mm = [[Axx, 0.0 , 1*Axz/vx_1], [0.0,0.0,0.0], [-1.*Azx/vz_0, 0.0, -1.5*Azz]]

#Mm = [[0.0, 0.0 , 0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]



#Paper plane initial conditions

avion1 = Paper_plane(x0=[0.0,0.0,0.0], v0=[vx_0,0.0,vz_0], M = Mm)

for i in range(0,1000):
    avion1.evol()

x1, y1, z1, vx1, vy1 ,vz1 = avion1.data()


#Graficator 2D

print(Mm)

plt.grid()
plt.xlabel("Distancia X [m]")
plt.ylabel("Distancia Z [m]")
plt.plot(x1, z1)
plt.plot(-M1L1[1], M1L1[2], color = "red")
plt.plot(-M1L2[1], M1L2[2], "-o", color = "red")
plt.plot(-M1L3[1], M1L3[2], "--", color = "red")
plt.plot(-M2L1[1], M2L1[2], color = "blue")
plt.plot(-M2L2[1], M2L2[2], "-o", color = "blue")
plt.plot(-M2L3[1], M2L3[2], "--", color = "blue")
#plt.plot(-M2L4[1], M2L4[2], "o", color = "blue")
#plt.plot(-M2L5[1], M2L5[2], "o", color = "blue")
plt.plot(M3L1[1], M3L1[2], color = "green")
plt.plot(-M3L2[1], M3L2[2], "-o", color = "green")
plt.plot(-M3L3[1], M3L3[2], "--", color = "green")


plt.savefig("Trayectorias.jpeg", dpi = 1200)
plt.show()



"""



"""

