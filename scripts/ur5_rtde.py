import rtde_control
import rtde_receive

# Connection
rtde_c = rtde_control.RTDEControlInterface("192.168.0.102")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.102")

pose = rtde_r.getActualTCPPose()
print(pose)


# MoveL
rtn = rtde_c.moveL([-0.292, -0.416, 0.488, 1.083, -2.948, 0], 0.5, 0.3)    # (pose, speed, acceleration, asynchronous: bool) -> bool

# Position
pose = rtde_r.getActualTCPPose()
print(pose)

# Force
force = rtde_r.getActualTCPForce()
print(force)
