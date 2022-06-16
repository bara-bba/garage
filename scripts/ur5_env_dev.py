import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

import rtde_control
import rtde_receive

import gym
from gym import utils, error, spaces
from gym.utils import seeding
import math

from garage import Environment, EnvSpec, EnvStep, StepType

# Mean Gaussian Distribution Noise UR5
MUX = 0.175
MUY = -0.118
MUZ = -0.508

# Connection
UR5IP = "192.168.0.102"
c = rtde_control.RTDEControlInterface(UR5IP)
r = rtde_receive.RTDEReceiveInterface(UR5IP)

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


def initialize_target():
    init_qpos = [5.74918477e-02, -3.74263917e-01, 2.4090970e-01, 3.00072744e+00, -9.26571906e-01, 2.10862988e-04]
    target = [5.74955744e-02, -3.74280655e-01, 2.09995799e-01, 3.00065647e+00, -9.26503897e-01, 1.99188108e-04]
    init_qvel = [0, 0, 0, 0, 0, 0]
    c.moveL(init_qpos, 0.05, TCPddmax)

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

    print("Aligning")

    q = r.getActualTCPPose()
    q_aliigned = (q[0], q[1], q[2], np.pi, 0, 0)

    print("Offset")

    c.moveL(q_aliigned, TCPdmax, TCPddmax)
    c.moveUntilContact([0, 0, -0.01, 0, 0, 0])
    offset = r.getActualTCPPose()[2]

    print("Moving")

    path = [q_path3, q_path2, q_path1, q_init]
    c.moveL(q_path3, vel, acc)
    c.moveL(q_path2, vel, acc)
    c.moveL(q_path1, vel, acc)
    c.moveL(q_init, vel, acc)

    print(f"Offset: {offset}")

    return offset

offset = 0.1990014370730782
# offset = align_offset([5.74918477e-02, -3.74263917e-01, 2.4090970e-01, 3.00072744e+00, -9.26571906e-01, 2.10862988e-04])
#
class UR5Env(Environment):
    """Real UR5 Environment Implementation"""

    def __init__(self, max_episode_length=math.inf):

        self.viewer = None
        self._viewers = {}
        self._visualize = False
        self._max_episode_length = max_episode_length

        c.zeroFtSensor()

        self.offset = offset
        self.init_qpos, self.init_qvel, target = initialize_target()
        self.target = np.asarray(target[:3]) - np.asarray((0, 0, self.offset))
        self.dist_max = 0.0006*300*np.sqrt(3)

        self.direction_init = R.from_rotvec(self.init_qpos[3:6])
        self.tcp_frame_init = self.direction_init.apply(ref_frame)
        self.tcp_pose_init = np.concatenate([self.init_qpos[:3], self.direction_init.as_rotvec()])

        self.diff_vector = np.array([0, 0, 0], np.float32)
        #


        self.metadata = {"render.modes": ["human"]}

        self.counter = 0

        self.set_action_space()

        action = self.action_space.sample()
        self.dp = np.zeros_like(action)
        self.dtheta = np.zeros_like(action[3:6])

        self.exec = 0
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)
        # print(self.observation_space)

        self.seed()

    def step(self, action):

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

        self.move(action)

        self.diff_vector = site_pose[:3] - self.target

        dist = np.linalg.norm(self.diff_vector)

        if dist < 0.005:  # Millimiters
            done = True
            print("DONE!!!!!!!!!!!!!")
            reward_done = 1

        else:
            done = False
            reward_done = 0

        f = r.getActualTCPForce()
        f = (f[0] - MUX, f[1] - MUY, f[2] - MUZ)

        force_v = np.linalg.norm(f[:3])
        torque_v = np.linalg.norm(f[3:6])
        # print(f"Force_Vector: {force_v}")

        reward_pos = - (dist/self.dist_max)**0.2

        if force_v > 20:
            done = True
            print("FAIL")
            reward_done = -1

        reward = reward_pos + reward_done

        self.counter += 1

        info = {}
        ob = self._get_obs()

        return ob, reward, done, info

    def move(self, action):
        self.dp += action
        xyz = R.from_euler('xyz', self.dp[3:6])
        self.dtheta = xyz.as_rotvec()
        pose = np.concatenate([self.dp[:3], self.dtheta])
        dp_to_pose = c.poseTrans(p_from=self.tcp_pose_init, p_from_to=pose)

        # print(f"Action: {action}")
        # print(f"Target pose: {pose}")
        c.moveL(dp_to_pose, TCPdmax, TCPddmax)

    def _get_obs(self):

        global ref_frame

        actual_pose = r.getActualTCPPose()

        direction = R.from_rotvec(actual_pose[3:6])
        direction_init = R.from_rotvec(self.init_qpos[3:6])

        tcp_frame = direction.apply(ref_frame)
        tcp_frame_init = direction_init.apply(ref_frame)

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
        f[0] = f[0] - MUX
        f[1] = f[1] - MUY
        f[2] = f[2] - MUZ

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

        # print(type(site_pose))
        # print(type(f))
        # print(type(self.diff_vector))

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
        self.counter = 0
        self.dp = np.zeros_like(self.init_qpos)

        c_xy = 0.05
        c_z = 0.01
        c_a = 0.1

        qpos = np.asarray(self.init_qpos)
        qpos[:2] = self.init_qpos[:2] + self.np_random.uniform(low=-c_xy, high=c_xy, size=2)
        qpos[2:3] = self.init_qpos[2:3] + self.np_random.uniform(low=-c_z, high=c_z, size=1)
        qpos[3:6] = self.init_qpos[3:6] + self.np_random.uniform(low=-c_a, high=c_a, size=3)

        c.zeroFtSensor()

        self.set_state(qpos)
        return self._get_obs()

    def reset(self):
        c.moveL(self.init_qpos, TCPdmax, TCPddmax)
        ob = self.reset_model()
        return ob, {}

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}
        print("Close")

    def _set_action_space(self):
        center = r.getActualTCPPose()
        offset = np.ones_like(center).astype(np.float32)
        low, high = -0.0006 * offset, 0.0006 * offset
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # print(f"Action Space: {self.action_space}")
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    @staticmethod
    def set_state(ctrl):
        c.moveL(ctrl, TCPdmax, TCPddmax)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
