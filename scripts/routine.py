import numpy as np
import rtde_control
import rtde_receive
from rtde_control import Path, PathEntry

from scipy.spatial.transform import Rotation as R

# Connection
c = rtde_control.RTDEControlInterface("192.168.0.102")
r = rtde_receive.RTDEReceiveInterface("192.168.0.102")

# INITIALIZATION
c.zeroFtSensor()
q_init = r.getActualTCPPose()
print(q_init)

q_aliigned = (q_init[0], q_init[1], q_init[2], np.pi, 0, 0)

c.moveL(q_aliigned, 0.05, 0.01)
c.moveUntilContact([0, 0, -0.01, 0, 0, 0])

offset = r.getActualTCPPose()[2]


print(offset)


print(c.getTCPOffset())

vel = 0.15
acc = 0.05
blend = 0.1

q_init = [5.74918477e-02, -3.74263917e-01, 2.4090970e-01, 3.00072744e+00, -9.26571906e-01, 2.10862988e-04, vel, acc, blend]
q_path1 = [ -2.95211376e-01, -3.24788527e-01, 3.08864492e-01, 3.13236319e+00, -2.40632021e-01, 5.33081548e-06, vel, acc, blend]
q_path2 = [-4.04960527e-01, 3.16603984e-02, 3.14693539e-01, -3.13717128e+00, 1.65833565e-01, 4.90767293e-05, vel, acc, blend]
q_path3 = [-2.56335904e-01, 2.50040996e-01, 2.32607645e-01, 3.11529306e+00, -4.05618169e-01, -7.38833067e-06, vel, acc, blend]

path = [q_path3, q_path2, q_path1, q_init]
c.moveL(path)

path = [q_init, q_path1, q_path2, q_path3]
c.moveL(path)

c.stopScript()
