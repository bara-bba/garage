import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class PandaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.counter = 0
        self.reward_force = 0
        mujoco_env.MujocoEnv.__init__(self, "/home/bara/PycharmProjects/garage/panda/insert_base.xml", 100)

        utils.EzPickle.__init__(self)

    def step(self, action):

        new_action = self.sim.data.qpos + action

        # print("ACTION: " + str(action))
        # print("NEW_ACTION: " + str(new_action))
        self.do_simulation(new_action, self.frame_skip)

        diff_vector = self.get_site_xpos("insert_site") - self.get_site_xpos("base_site")
        dist = np.linalg.norm(diff_vector)

        if dist < 0.005:  # Millimiters
            done = True
            reward_done = 100
        else:
            done = False
            reward_done = 0

        f = self.sim.data.sensordata.flat[:]

        force_v = np.linalg.norm(f[:3])
        torque_v = np.linalg.norm(f[3:6])

        if force_v > 30:
            done = True
            reward_done = -100

        self.reward_force += np.mean(force_v + torque_v)

        reward_pos = -dist*1.8
        reward = reward_pos + reward_done - self.reward_force

        self.counter += 1

        info = {}
        ob = self._get_obs()

        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:],
                self.sim.data.sensordata.flat[:],
                (self.get_site_xpos("insert_site") - self.get_site_xpos("base_site")).flat[:],
            ]
        ).astype(np.float32)

    def reset_model(self):
        c = 0.05
        self.counter = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.4

    def get_site_xpos(self, site_name):
        return self.data.get_site_xpos(site_name)
