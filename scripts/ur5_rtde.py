import numpy as np
import rtde_control
import rtde_receive

from scipy.spatial.transform import Rotation as R

# Connection
c = rtde_control.RTDEControlInterface("192.168.0.102")
r = rtde_receive.RTDEReceiveInterface("192.168.0.102")

low = -0.0
high = 0.0

origin_frame = [0, 0, 0, 0, 0, 0]

# def rot_to_xyz(rot):
#     r = R.from_rotvec(rot)
#     zyx = r.as_euler('xyz')
#
#     return zyx
#
# def zyx_to_matrix(zyx):
#     r = R.from_euler('xyz', zyx)
#     matrix = r.as_matrix()
#
#     return matrix
#
# def matrix_to_xyz(matrix):
#     r = R.from_matrix(matrix)
#     xyz = r.as_euler('xyz')
#
#     return xyz
#
# def matrix_to_rot(matrix):
#     r = R.from_matrix(matrix)
#     rot = r.as_rotvec()
#
#     return rot
#
# def xyz_to_rot(xyz):
#     r = R.from_euler('xyz', xyz)
#     rot = r.as_rotvec()
#
#     return rot

ref_frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

q_init = [0.0, 0.4, 0.4, 2.363, -2.068, 0.0] #[0.4, 0, 0.4, 2.363, -2.068 , 0.0]
c.moveL(q_init)



# Random ACTION
# action = np.random.uniform(low=low, high=high, size=q_init.__len__())

action = np.array([0.1, 0.1, 0.1, 0., 0., 0.])
xyz = R.from_euler('xyz', action[3:6])
print(xyz.as_euler('xyz'))

pose = r.getActualTCPPose()
pose[0:3] += action[0:3]
rot = R.from_rotvec(pose[3:6])

frame_rotated = xyz.apply(rot.apply(ref_frame))
rot = R.from_matrix(frame_rotated)
pose[3:6] = rot.as_rotvec()
print(frame_rotated)

c.moveL(pose)

pose = r.getActualTCPPose()

direction_init = R.from_rotvec(q_init[3:6])
direction = R.from_rotvec(pose[3:6])

print(f"direction_init: {direction_init.as_rotvec()}")
print(f"direction: {direction.as_rotvec()}")

tcp_frame_init = direction_init.apply(ref_frame)
tcp_frame = direction.apply(ref_frame)

tcp_pose_init = np.concatenate([q_init[:3], direction_init.as_rotvec()])
tcp_pose = np.concatenate([pose[:3], direction.as_rotvec()])

print(f"tcp_pose_init: {tcp_pose_init}")
print(f"tcp_pose: {tcp_pose}")

tcp_pose_inv = np.concatenate([-tcp_pose[:3], direction.inv().as_rotvec()])
print(f"tcp_pose_inv: {tcp_pose_inv}")

pose_trans = c.poseTrans(p_from=tcp_pose_inv, p_from_to=tcp_pose_init)

print(pose_trans)

rotation = R.from_rotvec(pose_trans[3:6])

pose = np.asarray(pose[:3]) - np.asarray(q_init[:3])
xyz = rotation.as_euler('xyz')

print(f"dP: {pose}")
print(f"Rotation XYZ: {xyz}")

# c.moveL(c.poseTrans(r.getActualTCPPose(), pose), 0.5, 0.1)
print("OK")

print(c.poseTrans(q_init, action))