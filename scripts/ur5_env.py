import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict

import rtde_control
import rtde_receive

import gym
from gym import utils, error, spaces
from gym.utils import seeding

UR5IP = "192.168.0.102"

# Connection
c = rtde_control.RTDEControlInterface(UR5IP)
r = rtde_receive.RTDEReceiveInterface(UR5IP)

TCPdmax, TCPddmax = 0.05, 0.01
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


def initialize_target():

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


def move(action):

    global ref_frame

    pose = r.getActualTCPPose()

    xyz = R.from_euler('xyz', action[3:6])
    rot = R.from_rotvec(pose[3:6])
    frame_rotated = xyz.apply(rot.apply(ref_frame))
    rot = R.from_matrix(frame_rotated)
    # print(f"Pose: {pose}")

    pose[0:3] += action[0:3]    #nOOOOOOOOOOOOOOOOOOOOOOOOOOO dio can, fai la transpose
    pose[3:6] = rot.as_rotvec()

    # print(f"Action: {action}")
    # print(f"Target pose: {pose}")
    c.moveL(pose, TCPdmax, TCPddmax)


class UR5Env(gym.Env, utils.EzPickle):
    """Real UR5 Environment Implementation"""
    def __init__(self):
        utils.EzPickle.__init__(self)

        c.zeroFtSensor()

        self.init_qpos, self.init_qvel, target = initialize_target()

        self.diff_vector = np.array([0, 0, 0], np.float32)
        self.offset = np.array([0, 0, 0.05], np.float32)

        self.metadata = {
            "render.modes": ["human"],
        }

        self.target = target[:3]

        self.counter = 0

        try:
            c.moveL(self.init_qpos)
            print("InitQPose: " + str(self.init_qpos))
            c.moveL(np.concatenate((self.init_qpos[:3] - self.offset, self.init_qpos[3:])))
            c.moveL(self.init_qpos)
            print("QPose: " + str(self.init_qpos))
        except:
            raise PermissionError

        self._set_action_space()

        action = self.action_space.sample()

        self.exec = 0
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)
        print(self.observation_space)

        self.seed()


    def step(self, action):

        print(f"action: {action}")

        tcp_pose = np.asarray(r.getActualTCPPose())
        move(action)

        self.diff_vector = tcp_pose[:3] - self.target

        dist = np.linalg.norm(self.diff_vector)

        if dist < 0.005:  # Millimiters
            done = True
            print("DONE!!!!!!!!!!!!!")
            reward_done = 100

        else:
            done = False
            reward_done = 0

        reward_pos = -dist * 1.8
        reward = reward_pos + reward_done

        self.counter += 1

        info = {}
        ob = self._get_obs()

        print(f"Reward : {reward}")

        return ob, reward, done, info

    def _get_obs(self):

        global ref_frame

        actual_pose = r.getActualTCPPose()

        direction = R.from_rotvec(actual_pose[3:6])
        direction_init = R.from_rotvec(self.init_qpos[3:6])

        tcp_frame = direction.apply(ref_frame)
        tcp_frame_init = direction_init.apply(ref_frame)

        tcp_pose_init = np.concatenate([self.init_qpos[:3], direction_init.as_rotvec()])
        tcp_pose = np.concatenate([actual_pose[:3], direction.as_rotvec()])
        tcp_pose_inv = np.concatenate([-tcp_pose[:3], direction.inv().as_rotvec()])

        pose_trans = c.poseTrans(p_from=tcp_pose_inv, p_from_to=tcp_pose_init)
        print(f"pose_trans: {pose_trans}")
        rotation = R.from_rotvec(pose_trans[3:6])

        xyz = rotation.as_euler('xyz')

        # speed = np.array(r.getActualTCPSpeed(), dtype=np.float32)
        pose = np.asarray(actual_pose[:3]) - np.asarray(self.init_qpos[:3])
        print(pose)

        force = np.array(r.getActualTCPForce(), dtype=np.float32)
        speed = np.zeros_like(force)
        print(np.concatenate([pose[:3], xyz]))
        print(f"self.diff_vector: {self.diff_vector}")
        print(r.getRobotStatus())

        obs = np.concatenate(
            [
                np.concatenate([pose, xyz]),
                speed, #???/DT???????????
                force,
                self.diff_vector,
            ]
        ).astype(np.float32)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.init_qpos.__len__())
        )
        return self._get_obs()

    def reset(self):
        c.moveL(self.init_qpos, TCPdmax, TCPddmax)
        ob = self.reset_model()
        return ob, {}

    def set_state(self, ctrl):
        state = np.array(r.getActualTCPPose(), np.float32)
        c.moveL(ctrl, TCPdmax, TCPddmax)

    def close(self):
        print("Disconnected")

    def _set_action_space(self):
        center = r.getActualTCPPose()
        offset = np.ones_like(center)
        low, high = -0.01*offset, 0.01*offset
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        print(f"Action Space: {self.action_space}")
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space




