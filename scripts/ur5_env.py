import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

import rtde_io
import rtde_control
import rtde_receive

import gym
from gym import utils, error, spaces
from gym.utils import seeding

# Mean Gaussian Distribution Noise UR5
MUX = 0.175
MUY = -0.118
MUZ = -0.508

# Connection
UR5IP = "192.168.0.102"
# io = rtde_io.RTDEIOInterface(UR5IP)
c = rtde_control.RTDEControlInterface(UR5IP)
r = rtde_receive.RTDEReceiveInterface(UR5IP)

print("Connected")

TCPdmax, TCPddmax = 0.1, 0.05
ref_frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def initialize_target_teach():
    c.teachMode()

    print(f"Move to final position. (Enter)")
    choice = None
    while choice != '':
        choice = input().lower()

    c.endTeachMode()

    target = r.getActualTCPPose()
    print(f"target: {target}")

    c.teachMode()

    print(f"Move to initial position. (Enter)")
    choice = None

    while choice != '':
        choice = input().lower()

    c.endTeachMode()

    init_qpos = r.getActualTCPPose()
    init_qvel = r.getActualTCPSpeed()
    print(f"qpos_init: {init_qpos}")

    return init_qpos, init_qvel, target


# init_qpos = [0.05649304156930342, -0.3758193992621653, 0.23777181518273438, -2.9876820047633994, 0.9229206730917303, 0.0027145060149046858]
# c.moveL(init_qpos)

def initialize_target():
    init_qpos = r.getActualTCPPose() #[0.05725087622580927, -0.3731227489723776, 0.242, -1.1947490794358302, -2.897899633003305, -0.00242631993292831]
    target = [0.05723768954291797, -0.37317080968906796, 0.21202079879304652, -1.1952487756507504, -2.897753284713638, -0.001665361313775619]
    # print(f"targetTCP: {target}")
    init_qvel = [0, 0, 0, 0, 0, 0]

    return init_qpos, init_qvel, target


def align_offset(q_init):

    print("Offset_routine")

    vel = 0.15
    acc = 0.5
    blend = 0.05

    print("Moving")

    q_path1 = [-2.95211376e-01, -3.24788527e-01, 4.08864492e-01, 3.13236319e+00, -2.40632021e-01, 5.33081548e-06]
    q_path2 = [-4.04960527e-01, 3.16603984e-02, 4.14693539e-01, -3.13717128e+00, 1.65833565e-01, 4.90767293e-05]
    q_path3 = [-2.56335904e-01, 2.50040996e-01, 2.32607645e-01, 3.11529306e+00, -4.05618169e-01, -7.38833067e-06]

    path = [q_init, q_path1, q_path2, q_path3]
    c.moveL(q_path1, vel, acc)
    c.moveL(q_path2, vel, acc)
    c.moveL(q_path3, vel, acc)

    sys.stdout.write('\033[2K\033[1G')
    print("Aligning")

    q = r.getActualTCPPose()
    q_aliigned = (q[0], q[1], q[2], np.pi, 0, 0)

    sys.stdout.write('\033[2K\033[1G')
    print("Offset")

    c.moveL(q_aliigned, TCPdmax, TCPddmax)
    c.moveUntilContact([0, 0, -0.01, 0, 0, 0])
    offset = r.getActualTCPPose()[2]

    sys.stdout.write('\033[2K\033[1G')
    print("Moving")

    path = [q_path3, q_path2, q_path1, q_init]
    c.moveL(q_path3, vel, acc)
    c.moveL(q_path2, vel, acc)
    c.moveL(q_path1, vel, acc)
    c.moveL(q_init, vel, acc)

    sys.stdout.write('\033[2K\033[1G')
    print(f"Offset: {offset}")

    return offset

offset =  0.18756524024506743
# offset = align_offset([5.74918477e-02, -3.74263917e-01, 2.4090970e-01, 3.00072744e+00, -9.26571906e-01, 2.10862988e-04])

class UR5Env(gym.Env, utils.EzPickle):
    """Real UR5 Environment Implementation"""

    def __init__(self):

        # self.viewer = None
        # self._viewers = {}
        utils.EzPickle.__init__(self)

        c.zeroFtSensor()

        self.offset = offset
        self.init_qpos, self.init_qvel, target = initialize_target()
        self.target = np.asarray(target) - np.asarray((0, 0, self.offset, 0, 0, 0))
        # print(f"targetSITE: {self.target}")
        self.dist_max = 0.0006*50*np.sqrt(3)

        self.direction_init = R.from_rotvec(self.init_qpos[3:6])
        self.tcp_frame_init = self.direction_init.apply(ref_frame)
        self.tcp_pose_init = np.concatenate([self.init_qpos[:3], self.direction_init.as_rotvec()])

        self.diff_vector = np.array([0, 0, 0], np.float32)

        self.metadata = {"render_modes": ["human"]}

        self.counter = 0

        self.set_action_space()
        action = self.action_space.sample()
        # self.dp = np.asarray(c.poseTrans(p_from=self.tcp_pose_init, p_from_to=self.tcp_pose_init))
        self.dp = np.zeros_like(action)
        self.dtheta = np.zeros_like(action[3:6])

        self.exec = 0
        observation, _reward, done, _info = self.step(action)
        assert not done
        self.set_observation_space(observation)

        self.seed()
        sys.stdout.write("\033[K")
        # print("END INIT")

    def step(self, action):

        # print("STEP")

        # print(f"action: {action}")

        tcp_pose = np.asarray(r.getActualTCPPose())
        tcp_to_site = np.asarray((0, 0, self.offset, 0, 0, 0))

        if len(c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)) != 0:
            site_pose = c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)
            # print(f"site_pose: {site_pose}")
        else:
            while len(c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)) == 0:
                time.sleep(1)
                site_pose = c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)

        self.diff_vector = site_pose - self.target
        self.diff_vector = self.diff_vector[:3]

        # init_to_pose = np.array(c.poseTrans(p_from=self.tcp_pose_init, p_from_to=tcp_pose))
        # print(f"init_to_pose: {init_to_pose}")
        # init_to_target = np.array(c.poseTrans(p_from=self.tcp_pose_init, p_from_to=self.target))
        # print(f"init_to_target: {init_to_target}")


        # print(f"diff_vector: {self.diff_vector}")

        dist = np.linalg.norm(self.diff_vector)
        # print(f"dist: {dist}")

        if dist < 0.005:  # Millimiters
            print("DONE!!!!!!!!!!!!!")
            # io.setStandardDigitalOut(1, not(r.getActualDigitalOutputBits()))
            self.move(np.concatenate([[0, 0, 1e-2], [0, 0, 0]]))
            done = True
            reward_done = 1
            # c.stopScript()
            # c.disconnect()

        else:
            done = False
            reward_done = 0


        f = r.getActualTCPForce()
        # f = (f[0] - MUX, f[1] - MUY, f[2] - MUZ)

        force_v = np.linalg.norm(f[:3])
        torque_v = np.linalg.norm(f[3:6])
        # print(f"Force_Vector: {force_v}")

        if self.diff_vector[0] <= 0.001 and self.diff_vector[1] <= 0.001 and self.counter > 40 and self.counter % 3 != 0:
            action = np.concatenate([[0, 0, 1e-3], [0, 0, 0]])

        if force_v > 15:
            reward_force = -1 / 300
            # reward_force = 0
            if len(action) != 0:
                self.move(-3*action)
                diff_direction = self.diff_vector/np.linalg.norm(self.diff_vector)
                diff_direction = np.concatenate([diff_direction, r.getActualTCPPose()[3:6]])
                # print(f"diff_direcrtion: {diff_direction}")
                dir = np.concatenate([(-self.diff_vector[:2]/dist*0.8e-3), [0, 0, 0, 0]])
                self.move(-dir)

        else:
            reward_force = 0

        if force_v > 20:
            done = True
            reward_done = -1
            print("Too much force")

        if self.counter % 3 == 0:
            action = np.concatenate([(self.diff_vector / dist * 1e-3), [0, 0, 0]])

        self.move(action)

        reward_pos = - (dist/self.dist_max)**0.2

        reward = reward_pos + reward_done + reward_force

        self.counter += 1
        # print(f"reward: {reward}")

        info = {}
        ob = self._get_obs()

        # print("END STEP")

        return ob, reward, done, info

    def move(self, action):
        # print("MOVE")
        # print(f"action: {action}")
        self.dp += action

        xyz = R.from_euler('xyz', self.dp[3:6])
        self.dtheta = xyz.as_rotvec()
        pose = np.concatenate([self.dp[:3], self.dtheta])
        dp_to_pose = c.poseTrans(p_from=self.tcp_pose_init, p_from_to=pose)

        # print(f"Action: {action}")
        # print(f"Target pose: {pose}")

        c.moveL(dp_to_pose, TCPdmax, TCPddmax)
        # print("DONE MOVE")

    def _get_obs(self):
        # print("GET_OBS")

        global ref_frame

        actual_pose = r.getActualTCPPose()

        direction = R.from_rotvec(actual_pose[3:6])
        direction_init = R.from_rotvec(self.init_qpos[3:6])

        # tcp_frame = direction.apply(ref_frame)
        # tcp_frame_init = direction_init.apply(ref_frame)

        tcp_pose_init = np.concatenate([self.init_qpos[:3], direction_init.as_rotvec()])
        tcp_pose_init_inv = np.concatenate([-direction_init.inv().apply(tcp_pose_init[:3]), direction_init.inv().as_rotvec()])
        tcp_pose = np.concatenate([actual_pose[:3], direction.as_rotvec()])

        if len(c.poseTrans(p_from=tcp_pose_init_inv, p_from_to=tcp_pose)) != 0:
            pose_trans = c.poseTrans(p_from=tcp_pose_init_inv, p_from_to=tcp_pose)
        else:
            pose_trans = np.random.uniform(low=-0.000001, high=0.000001, size=6)
        # print(f"pose_trans: {pose_trans}")
        rotation = R.from_rotvec(pose_trans[3:6])
        xyz = rotation.as_euler('xyz')

        pose = np.concatenate([pose_trans[:3], xyz])
        f = np.array(r.getActualTCPForce(), dtype=np.float32)
        # f[0] = f[0] - MUX
        # f[1] = f[1] - MUY
        # f[2] = f[2] - MUZ

        tcp_to_site = np.concatenate([[0, 0, self.offset], [0, 0, 0]])

        if len(c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)) != 0:
            site_pose = c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)
            # print(f"site_pose: {site_pose}")
        else:
            while len(c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)) == 0:
                time.sleep(1)
                site_pose = c.poseTrans(p_from=tcp_pose, p_from_to=tcp_to_site)

        # print(f"self.diff_vector: {self.diff_vector}")
        site_pose = np.array(site_pose, dtype=np.float32)

        obs = np.concatenate(
            [
                site_pose.flat[:],
                f.flat[:],
                self.diff_vector.flat[:],
            ]
        ).astype(np.float32)

        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        # print("RESET_MODEL")
        self.counter = 0

        c_xy = 0.01
        c_z = 0.00
        c_a = 0.0

        qpos = np.asarray(self.init_qpos)
        qpos[:2] = self.init_qpos[:2] + np.random.uniform(low=-c_xy, high=c_xy, size=2)
        qpos[2:3] = self.init_qpos[2:3] + np.random.uniform(low=-c_z, high=c_z, size=1)
        qpos[3:6] = self.init_qpos[3:6] + np.random.uniform(low=-c_a, high=c_a, size=3)

        # print("RESET_STRANGE")
        now_pose = r.getActualTCPPose()
        now_pose_high = [now_pose[0], now_pose[1], 0.30, now_pose[3], now_pose[4], 0]
        qpos_high = [qpos[0], qpos[1], 0.30, qpos[3], qpos[4], qpos[5]]
        c.moveL(now_pose_high, TCPdmax * 3, TCPddmax * 3)
        c.moveL(qpos_high, TCPdmax * 3, TCPddmax * 3)
        c.moveL(qpos, TCPdmax * 3, TCPddmax * 3)

        self.dp = np.zeros_like(self.init_qpos)
        self.dtheta = np.zeros(3)

        c.zeroFtSensor()

        return self._get_obs()

    def reset(self):
        print("RESET")

        ob = self.reset_model()
        return ob

    # @property
    # def close(self):
    #     if self.viewer is not None:
    #         # self.viewer.finish()
    #         self.viewer = None
    #         self._viewers = {}
    #     print("Close")
    #     return

    def set_action_space(self):
        center = r.getActualTCPPose()
        offset = np.ones_like(center).astype(np.float32)
        low, high = -0.0006 * offset, 0.0006 * offset
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # print(f"Action Space: {self.action_space}")
        return self.action_space

    def set_observation_space(self, observation):
        # print(f"observation: {observation}")
        self.observation_space = convert_observation_to_space(observation)
        # print(f"observationSpace: {self.observation_space}")
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
