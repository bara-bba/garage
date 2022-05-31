import numpy as np
from scipy import fftpack
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numpy import genfromtxt

ur_force = genfromtxt("/home/bara/drp/Dropbox/Camozzi-POLIMI project weekly meetings/Lorenzo Barattin/ur5_force.csv",
                delimiter=',')
ur_torque = genfromtxt("/home/bara/drp/Dropbox/Camozzi-POLIMI project weekly meetings/Lorenzo Barattin/ur5_torque6.csv",
                delimiter=',')
panda = genfromtxt("/home/bara/PycharmProjects/garage/panda_force.csv",
                   delimiter=',')

time_step = 1/2000

ur0 = ur_force[:, 0]
ur1 = ur_force[:, 1]
ur2 = ur_force[:, 2]
ur3 = ur_torque[:, 0]
ur4 = ur_torque[:, 1]
ur5 = ur_torque[:, 2]

panda0 = panda[:, 0]
panda1 = panda[:, 1]
panda2 = panda[:, 2]


UR0 = fftpack.fft(ur0)
UR1 = fftpack.fft(ur1)
UR2 = fftpack.fft(ur2)
UR3 = fftpack.fft(ur3)
UR4 = fftpack.fft(ur4)
UR5 = fftpack.fft(ur5)

PANDA0 = fftpack.fft(panda0)
PANDA1 = fftpack.fft(panda1)
PANDA2 = fftpack.fft(panda2)


t0 = np.arange(0, len(ur0), 1)
t1 = np.arange(0, len(ur1), 1)
t2 = np.arange(0, len(ur2), 1)
t3 = np.arange(0, len(ur3), 1)
t4 = np.arange(0, len(ur4), 1)
t5 = np.arange(0, len(ur5), 1)

pt0 = np.arange(0, len(panda0), 1)
pt1 = np.arange(0, len(panda1), 1)
pt2 = np.arange(0, len(panda2), 1)


N0 = len(UR0)
N1 = len(UR1)
N2 = len(UR2)
N3 = len(UR3)
N4 = len(UR4)
N5 = len(UR5)

n0 = np.arange(N0)
n1 = np.arange(N1)
n2 = np.arange(N2)
n3 = np.arange(N3)
n4 = np.arange(N4)
n5 = np.arange(N5)


frequency0 = fftpack.fftfreq(ur0.size, d=time_step)
frequency1 = fftpack.fftfreq(ur1.size, d=time_step)
frequency2 = fftpack.fftfreq(ur2.size, d=time_step)
frequency3 = fftpack.fftfreq(ur3.size, d=time_step)
frequency4 = fftpack.fftfreq(ur4.size, d=time_step)
frequency5 = fftpack.fftfreq(ur5.size, d=time_step)

pfrequency0 = fftpack.fftfreq(panda0.size, d=time_step)
pfrequency1 = fftpack.fftfreq(panda1.size, d=time_step)
pfrequency2 = fftpack.fftfreq(panda2.size, d=time_step)

fig0 = plt.figure(figsize=(20, 10))
grid0 = plt.GridSpec(3, 9, wspace=0.5)

plt.suptitle("UR5 Noise Spectrum")

# Force Noise
plt.subplot(grid0[0, :-1])
plt.title("Fx")
plt.stem(frequency0[1:], np.abs(UR0)[1:], "#d45769", label="x-signal", markerfmt=" ")
plt.subplot(grid0[1, :-1])
plt.title("Fy")
plt.stem(frequency1[1:], np.abs(UR1)[1:], "#e69d45", label="y-signal", markerfmt=" ")
plt.subplot(grid0[2, :-1])
plt.title("Fz")
plt.stem(frequency2[1:], np.abs(UR2)[1:], "#308695", label="z-signal", markerfmt=" ")


plt.subplot(grid0[0, -1])
plt.axis('off')
plt.plot(ur0[:1000], t0[:1000], color="#d45769", alpha=1, label="x")
plt.subplot(grid0[1, -1])
plt.axis('off')
plt.plot(ur1[:1000], t1[:1000], color="#e69d45", alpha=1, label="y")
plt.subplot(grid0[2, -1])
plt.axis('off')
plt.plot(ur2[:1000], t2[:1000], color="#308695", alpha=1, label="z")

fig1 = plt.figure(figsize=(20, 10))
grid1 = plt.GridSpec(3, 9, wspace=0.5)

plt.suptitle("UR5 Noise Spectrum")

# Torque Noise
plt.subplot(grid1[0, :-1])
plt.title("Rx")
plt.stem(frequency3[1:], np.abs(UR3)[1:], "#d45769", label="Rx-signal", markerfmt=" ")
plt.subplot(grid1[1, :-1])
plt.title("Ry")
plt.stem(frequency4[1:], np.abs(UR4)[1:], "#e69d45", label="Ry-signal", markerfmt=" ")
plt.subplot(grid1[2, :-1])
plt.title("Rz")
plt.stem(frequency5[1:], np.abs(UR5)[1:], "#308695", label="Rz-signal", markerfmt=" ")


plt.subplot(grid1[0, -1])
plt.axis('off')
plt.plot(ur3[:1000], t3[:1000], color="#d45769", alpha=1, label="x")
plt.subplot(grid1[1, -1])
plt.axis('off')
plt.plot(ur4[:1000], t4[:1000], color="#e69d45", alpha=1, label="y")
plt.subplot(grid1[2, -1])
plt.axis('off')
plt.plot(ur5[:1000], t5[:1000], color="#308695", alpha=1, label="z")


fig2 = plt.figure(figsize=(20, 10))
grid2 = plt.GridSpec(3, 9, wspace=0.5)

plt.suptitle("MuJoCo Noise Spectrum")

# Force Noise
plt.subplot(grid2[0, :-1])
plt.title("Fx")
plt.stem(pfrequency0[1:], np.abs(PANDA0)[1:], "#d45769", label="x-signal", markerfmt=" ")
plt.subplot(grid2[1, :-1])
plt.title("Fy")
plt.stem(pfrequency1[1:], np.abs(PANDA1)[1:], "#e69d45", label="y-signal", markerfmt=" ")
plt.subplot(grid2[2, :-1])
plt.title("Fz")
plt.stem(pfrequency2[1:], np.abs(PANDA2)[1:], "#308695", label="z-signal", markerfmt=" ")


plt.subplot(grid2[0, -1])
plt.axis('off')
plt.plot(panda0[1000:2000], pt0[:1000], color="#d45769", alpha=1, label="x")
plt.subplot(grid2[1, -1])
plt.axis('off')
plt.plot(panda0[1000:2000], pt1[:1000], color="#e69d45", alpha=1, label="y")
plt.subplot(grid2[2, -1])
plt.axis('off')
plt.plot(panda0[1000:2000], pt2[:1000], color="#308695", alpha=1, label="z")


fig3 = plt.figure(figsize=(20, 10))
grid3 = plt.GridSpec(3, 9, wspace=0.5)

plt.suptitle("MuJoCo Noise Spectrum")

plt.show()
