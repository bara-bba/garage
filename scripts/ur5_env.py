import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict

import rtde_control
import rtde_receive

import gym
from gym import utils, error, spaces
from gym.utils import seeding

import csv
csv_file = open("check.csv", 'w')

csv_writer = csv.writer(csv_file, delimiter=",")

UR5IP = "192.168.0.102"

# Connection
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
        low = np.full(observation.shape, -float("inf"))
        high = np.full(observation.shape, float("inf"))
        space = spaces.Box(low, high, dtype=np.float32)
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
    init_qpos = [-0.36223411523660765, 0.0731373793667136, 0.25290876626161135, -2.8751659715738795, -1.2103758598038512, 0.03565843724117852]
    target = [-0.3647102079578835, 0.06622577877285467, 0.2129475683324661, -2.908459614606202, -1.0880204354463259, -0.020409988674964897]
    init_qvel = [0, 0, 0, 0, 0, 0]
    c.moveL(init_qpos, 0.01, TCPddmax)

    return init_qpos, init_qvel, target


class UR5Env(gym.Env, utils.EzPickle):
    """Real UR5 Environment Implementation"""
    def __init__(self):
        utils.EzPickle.__init__(self)

        self.csv_content = ['counter', 'x', 'y', 'z', 'obs_x', 'obs_y', 'obs_z', 'reward']
        # print(r.getActualTCPForce())
        c.zeroFtSensor()
        # print(r.getActualTCPForce())

        self.init_qpos, self.init_qvel, target = initialize_target()

        self.direction_init = R.from_rotvec(self.init_qpos[3:6])
        self.tcp_frame_init = self.direction_init.apply(ref_frame)
        self.tcp_pose_init = np.concatenate([self.init_qpos[:3], self.direction_init.as_rotvec()])

        self.diff_vector = np.array([0, 0, 0], np.float32)
        self.offset = np.array([0, 0, 0.05], np.float32)

        self.metadata = {
            "render.modes": ["human"],
        }

        self.target = np.asarray(target[:3])

        self.counter = 0
        #
        # try:
        #     c.moveL(self.init_qpos)
        #     print("InitQPose: " + str(self.init_qpos))
        #     c.moveL(np.concatenate((self.init_qpos[:3] - self.offset, self.init_qpos[3:])))
        #     c.moveL(self.init_qpos)
        #     print("QPose: " + str(self.init_qpos))
        # except:
        #     raise PermissionError

        self._set_action_space()

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
        self.move(action)

        self.diff_vector = tcp_pose[:3] - self.target

        dist = np.linalg.norm(self.diff_vector)

        if dist < 0.01:  # Millimiters
            done = True
            print("DONE!!!!!!!!!!!!!")
            reward_done = 100
            self.close()

        else:
            done = False
            reward_done = 0

        f = r.getActualTCPForce()

        force_v = np.linalg.norm(f[:3])
        torque_v = np.linalg.norm(f[3:6])
        # print(force_v)

        reward_pos = -dist * 1.8

        if force_v > 30:
            done = True
            reward_done = -100

        reward = reward_pos + reward_done

        self.counter += 1

        info = {}
        ob = self._get_obs()

        new_row = (self.counter, r.getActualTCPPose()[0], r.getActualTCPPose()[1], r.getActualTCPPose()[2], ob[0], ob[1], ob[2], reward)
        self.csv_content = np.vstack((self.csv_content, new_row))

        # print(f"Reward : {reward/1.8*100}")

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

        pose_trans = c.poseTrans(p_from=tcp_pose_init_inv, p_from_to=tcp_pose)
        # print(f"pose_trans: {pose_trans}")
        rotation = R.from_rotvec(pose_trans[3:6])
        xyz = rotation.as_euler('xyz')

        pose = np.concatenate([pose_trans[:3], xyz])
        # speed = np.array(r.getActualTCPSpeed(), dtype=np.float32)
        speed = np.zeros_like(pose)
        force = np.array(r.getActualTCPForce(), dtype=np.float32)

        # print(f"self.diff_vector: {self.diff_vector}")

        obs = np.concatenate(
            [
                pose,
                speed,
                force*10,
                self.diff_vector,
            ]
        ).astype(np.float32)
        # print(obs)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        c = 0.01
        self.dp = np.zeros_like(self.init_qpos)
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.init_qpos.__len__())
        )
        return self._get_obs()

    def reset(self):
        c.moveL(self.init_qpos, TCPdmax, TCPddmax)
        ob = self.reset_model()
        return ob, {}

    def set_state(self, ctrl):
        c.moveL(ctrl, TCPdmax, TCPddmax)

    def close(self):
        c.moveL(self.init_qpos)

        # ROW WRITER FOR check.csv
        # for row in self.csv_content:
        #     csv_writer.writerow(row)

        csv_file.close()
        c.disconnect()
        print("Disconnected")

    def _set_action_space(self):
        center = r.getActualTCPPose()
        offset = np.ones_like(center)
        low, high = -0.0006*offset, 0.0006*offset
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # print(f"Action Space: {self.action_space}")
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space


