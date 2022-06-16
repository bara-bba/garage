import numpy as np
import rtde_control
import csv
import rtde_receive
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

# Connection
c = rtde_control.RTDEControlInterface("192.168.0.102")
r = rtde_receive.RTDEReceiveInterface("192.168.0.102")

MAX = int(1e9)

c.zeroFtSensor()

counter = 0
i, j, k = 0, 0, 0

# x = np.zeros(MAX)
# y = np.zeros(MAX)
# z = np.zeros(MAX)
# rx = np.zeros(MAX)
# ry = np.zeros(MAX)
# rz = np.zeros(MAX)
# force = np.zeros(MAX)
# torque = np.zeros(MAX)
# time = np.zeros(MAX)
# result_x = np.zeros(MAX)
# result_y = np.zeros(MAX)
# result_z = np.zeros(MAX)

x = 0

origin_frame_pose = [0, 0, 0, 0, 0, 0]
ref_frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

q_init = r.getActualTCPPose()
f = np.asarray(q_init)
direction_init = R.from_rotvec(q_init[3:6])
tcp_frame_init = direction_init.apply(ref_frame)

with open('ur5_force.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["counter", "x", "y", "z", "rx", "ry", "rz", "force", "torque"])

    while counter < MAX:
        x_old = x
        f[:3] = direction_init.apply(r.getActualTCPForce()[:3])
        f[3:6] = direction_init.apply(r.getActualTCPForce()[3:6])
        x = f[0]
        y = f[1]
        z = f[2]
        rx = f[3]
        ry = f[4]
        rz = f[5]
        force = np.linalg.norm(f[:3])
        torque = np.linalg.norm(f[3:6])
        # if x[counter] != x[counter-1]:
        #     result_x[i] = x[counter]
        #     i += 1
        #
        # if y[counter] != y[counter-1]:
        #     result_y[j] = y[counter]
        #     j += 1
        #
        # if z[counter] != z[counter-1]:
        #     result_z[k] = z[counter]
        #     k += 1

        counter += 1
        # y_old = y
        # z_old = z
        # rx_old = rx
        # ry_old = ry
        # rz_old = rz

        if counter % 1e7 == 0:
            print(f"{counter/MAX*100}%")

        if x != x_old:
            csvwriter.writerow([counter, x, y, z, rx, ry, rz, force, torque])


csvfile.close()

# plt.subplot(2, 4, 1)
# plt.plot(time, x)
#
# plt.subplot(2, 4, 2)
# plt.plot(time, y)
#
# plt.subplot(2, 4, 3)
# plt.plot(time, z)
#
# plt.subplot(2, 4, 4)
# plt.plot(time, rx)
#
# plt.subplot(2, 4, 5)
# plt.plot(time, ry)
#
# plt.subplot(2, 4, 6)
# plt.plot(time, rz)
#
# plt.subplot(2, 4, 7)

print(np.mean(force))
