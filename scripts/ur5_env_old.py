import numpy as np
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

import gym
from gym import utils
from gym import error, spaces
from gym.utils import seeding

import rtde_control
import rtde_receive

DEFAULT_SIZE = 500

# Connection
c = rtde_control.RTDEControlInterface("192.168.0.102")
r = rtde_receive.RTDEReceiveInterface("192.168.0.102")

TCPdmax, TCPddmax = 0.05, 0.01


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

class UR5Env(gym.Env, utils.EzPickle):
    """Real UR5 Environment Implementation"""

    def __init__(self):
        utils.EzPickle.__init__(self)

        self.diff_vector = np.array([0.0, 0.0, 0.0], np.float32)
        print(self.diff_vector.shape)
        self.counter = 0

        self.offset = np.array([0, 0, 0.05, 0, 0, 0], np.float32)
        self.init_qpos = r.getActualTCPPose()           #ROT_VECTOR
        self.init_qvel = r.getActualTCPSpeed()

        try:
            c.moveL(self.init_qpos)
            print("InitQPose: " + str(self.init_qpos))
            c.moveL(self.init_qpos - self.offset)
            c.moveL(self.init_qpos + self.offset)
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

        self.exec = 1

        self.seed()

    def _set_action_space(self):
        center = r.getActualTCPPose()
        offset = np.ones_like(center)
        low, high = center - offset*0.05, center + offset*0.05
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        print(self.action_space)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space


    def step(self, action):

        # STEP
        # print("STATE" + str(self.data.qpos.astype(np.float32)))
        # print("ACTION" + str(action))

        # print(self.counter)
        # self.counter += 1
        target = (self.init_qpos - self.offset)
        TCPPose = r.getActualTCPPose()

        # print("TCPPose: " + str(TCPPose))

        rot_vec = R.from_rotvec(TCPPose[2:5])
        xyz_vec = rot_vec.as_euler('xyz')

        TCPPose[3] = xyz_vec[0]
        TCPPose[4] = xyz_vec[1]
        TCPPose[5] = xyz_vec[2]

        # print("TCPPoseXYZ: " + str(TCPPose))

        new_action = r.getActualTCPPose() + action

        print("ACTION: " + str(action))
        print("NEW_ACTION: " + str(new_action))

        if self.exec == 1:
            self.do_simulation(new_action)


        self.diff_vector[0] = TCPPose[0] - target[0]
        self.diff_vector[1] = TCPPose[1] - target[1]
        self.diff_vector[2] = TCPPose[2] - target[2]

        dist = np.linalg.norm(self.diff_vector)

        # REWARD
        if dist < 0.005:
            reward_pos = 100
            done = True
        else:
            reward_pos = 0
            done = False

        reward_dist = -dist
        # reward_action = -self.counter/20
        reward = 1*reward_dist + reward_pos       # More contributions to rewards may be added

        ob = self._get_obs()

        return ob, reward, done, {}

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=6)
        )
        return self._get_obs()

    def _get_obs(self):
        # print("TCPPose: " + str(r.getActualTCPPose()))
        # print("TCPPoseType: " + str(np.array(r.getActualTCPPose(), np.float32)))
        return np.concatenate(
            [
                np.array(r.getActualTCPPose(), dtype=np.float32).flat,
                np.array(r.getActualTCPSpeed(), dtype=np.float32).flat,
                np.array(r.getActualTCPForce(), dtype=np.float32).flat,
                self.diff_vector.flat,
            ]
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        c.moveL(self.init_qpos, TCPdmax, TCPddmax)
        ob = self.reset_model()
        print(ob)
        return ob, {}

    def set_state(self, ctrl):
        state = np.array(r.getActualTCPPose(), np.float32)
        c.moveL(ctrl, TCPdmax, TCPddmax)

    def do_simulation(self, ctrl):

        xyz_vec = R.from_euler('xyz', ctrl[3:6])
        rot_vec = xyz_vec.as_rotvec()

        ctrl[3] = rot_vec[0]
        ctrl[4] = rot_vec[1]
        ctrl[5] = rot_vec[2]

        ok = c.moveL(ctrl, TCPdmax, TCPddmax)
        print("CTRL: " + str(ctrl))
        return ok

    def get_joint_xpos(self):
        return r.getActualQ()

    def state_vector(self):
        return np.concatenate([r.getActualTCPPose(), r.getActualTCPSpeed()])


