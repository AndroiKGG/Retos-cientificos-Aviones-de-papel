"Paper plane code"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class Paper_plane():

    def __init__(self, x0 = [0.0,0.0,0.0], v0 = [0.0,0.0,0.0], dt = 0.01):
        self.x = torch.tensor(x0)
        self.v = torch.tensor(v0)
        self.storex = [x0[0]]
        self.storey = [x0[1]]
        self.storez = [x0[2]]
        self.storevx = [v0[0]]
        self.storevy = [v0[1]]
        self.storevz = [v0[2]]
        self.dt = dt
    
    def Matrix(self, A = torch.zeros(9).reshape(3,3), Ext = False):
        if Ext:
            self.M = A
            return self.M
        else:
            self.M = torch.ones(9).reshape(3,3)/10
        return self.M
    
    def force(self, x, v):
        return torch.tensor([0, 0, -9.8])  + torch.matmul(self.M, x)

    
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
            self.Matrix()
            self.x, self.v = self.RK4(self.force, self.velocity, self.x, self.v)
            self.storex.append(self.x[0])
            self.storey.append(self.x[1])
            self.storez.append(self.x[2])
            self.storevx.append(self.v[0])
            self.storevy.append(self.v[1])
            self.storevz.append(self.v[2])

    
    def data(self):
        return self.storex, self.storey, self.storez, self.storevx, self.storevy, self.storevz
    
avion1 = Paper_plane(v0=[2,0,2])

for i in range(0,1000):
    avion1.evol()

x1, y1, z1, vx1, vy1 ,vz1 = avion1.data()

ax = plt.axes(projection ='3d')
ax.plot3D(x1, y1, z1)
plt.show()

"""


"""

