import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.spatial.transform import Rotation as R

def quiver(o, x, color='black'):
    return ax.quiver(o[0], o[1], o[2], x[0], x[1], x[2], arrow_length_ratio=0.1, color=color)

# def vectors
x_world = [1, 0, 0]
y_world = [0, 1, 0]
z_world = [0, 0, 1]

ref_x = [1, 0, 0]
ref_y = [0, 1, 0]
ref_z = [0, 0, 1]

rot = R.from_rotvec([3.137, -0.168, 0])
xyz = R.from_euler('xyz', [0, 0.5, 0])

new_x = rot.apply(xyz.apply(ref_x))
new_y = rot.apply(xyz.apply(ref_y))
new_z = rot.apply(xyz.apply(ref_z))

a = (ref_x, ref_y, ref_z)
b = (new_x, new_y, new_z)

result = R.align_vectors(a, b)
result[0].as_matrix()
print(result[0].as_rotvec())

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlim([-0.5, 1])
ax.set_ylim([-0.5, 1])
ax.set_zlim([-0.5, 1])
t = 0
o = [0, 0, 0]
tcp = [0, 0.4, 0.4]

def set_u(t):
    x = 0
    y = 0
    z = 0
    u = 1*t
    v = 1*t
    w = 1*t
    return x, y, z, u, v, w

quiver(o, x_world, color='r')
quiver(o, y_world, color='g')
quiver(o, z_world, color='b')

fx = quiver(tcp, new_x, color='r')
fy = quiver(tcp, new_y, color='g')
fz = quiver(tcp, new_z, color='b')

quiver = ax.quiver(*set_u(t))

def update(t):
    global fx
    global fy
    global fz
    t += 1
    # fx.remove()
    # fy.remove()
    # fz.remove()
    # quiver = ax.quiver(*set_u(t))

ani = FuncAnimation(fig, update, frames=np.linspace(0,2*np.pi,200), interval=50)
plt.show()