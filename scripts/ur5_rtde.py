import numpy as np
import rtde_control
import rtde_receive

from scipy.spatial.transform import Rotation as R

# Connection
c = rtde_control.RTDEControlInterface("192.168.0.102")
r = rtde_receive.RTDEReceiveInterface("192.168.0.102")

low = -0.0
high = 0.0

offset = 0.17484123

origin_frame_pose = [0, 0, 0, 0, 0, 0]
ref_frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# INITIALIZATION
# q_init = [0.0, 0.4, 0.4, 2.363, -2.068, 0.0] #[0.4, 0, 0.4, 2.363, -2.068 , 0.0]
q_init = [5.74918477e-02, -3.74263917e-01, 2.4090970e-01, 3.00072744e+00, -9.26571906e-01, 2.10862988e-04]
q_target = [5.74955744e-02, -3.74280655e-01, 2.09995799e-01, 3.00065647e+00, -9.26503897e-01, 1.99188108e-04]
q_path1 = [ -2.95211376e-01, -3.24788527e-01, 3.08864492e-01, 3.13236319e+00, -2.40632021e-01, 5.33081548e-06]
q_path2 = [-4.04960527e-01, 3.16603984e-02, 3.14693539e-01, -3.13717128e+00, 1.65833565e-01, 4.90767293e-05]
q_path3 = [-2.56335904e-01, 2.50040996e-01, 2.32607645e-01, 3.11529306e+00, -4.05618169e-01, -7.38833067e-06]
# q_init = r.getActualTCPPose()
direction_init = R.from_rotvec(q_init[3:6])
tcp_frame_init = direction_init.apply(ref_frame)
tcp_pose_init = np.concatenate([q_init[:3], direction_init.as_rotvec()])

rot_init = R.from_rotvec(q_init[3:6])
# c.moveL(q_target, 0.05, 0.01)
# c.moveL(q_init, 0.05, 0.01)

# ACTION
action = np.array([-0, 0, 0.01, 0., 0, 0])

# ACTION FRAME CONVERSION
xyz = R.from_euler('xyz', action[3:6])
frame_rotated_xyz = xyz.apply(rot_init.apply(ref_frame))
rot_xyz = R.from_matrix(frame_rotated_xyz)

# ACTION IMPLEMENTATION
dp = np.zeros_like(action)
dp[:3] = action[:3]                            #[dx, dy, dz, drx, dry, drz] wrt tcp_pose_init
dp[3:6] = xyz.as_rotvec()

dp_to_pose = c.poseTrans(p_from=tcp_pose_init, p_from_to=dp)
# c.moveL(dp_to_pose, 0.05, 0.01)

# CHECK POSE
pose = r.getActualTCPPose()
print(f"pose: {pose}")

direction = R.from_rotvec(pose[3:6])

# print(f"direction_init: {direction_init.as_rotvec()}")
# print(f"direction: {direction.as_rotvec()}")

tcp_frame = direction.apply(ref_frame)
tcp_pose = np.concatenate([pose[:3], direction.as_rotvec()])

# print(f"offset_vector: {direction.apply([0, 0, offset])}")

tcp_to_site = np.concatenate([[0, 0, offset], [0, 0, 0]])
site_pose = c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)
# print(f"site_pose: {site_pose}")
#
# print(f"tcp_pose_init: {tcp_pose_init}")
# print(f"tcp_pose: {tcp_pose}")
#
tcp_pose_init_inv = np.concatenate([-direction_init.inv().apply(tcp_pose_init[:3]), direction_init.inv().as_rotvec()])
# print(f"tcp_pose_init_inv: {tcp_pose_init_inv}")

pose_trans = c.poseTrans(p_from=tcp_pose_init_inv, p_from_to=tcp_pose)
print(f"pose trans: {pose_trans}")

rotation = R.from_rotvec(pose_trans[3:6])
xyz = rotation.as_euler('xyz')

# print(f"dP: {pose}")
# print(f"Rotation XYZ: {xyz}")
#
# print("OK")

c.stopScript()