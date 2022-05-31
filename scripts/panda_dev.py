import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


class PandaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.counter = 0
        self.reward_force = 0
        self.reward_x = 0
        self.reward_y = 0
        self.force_sum = 0
        self.dist_max = 0.0006*300*np.sqrt(3)
        mujoco_env.MujocoEnv.__init__(self, "/home/bara/PycharmProjects/garage/panda/insert_base.xml", 100)

        utils.EzPickle.__init__(self)

    def step(self, action):

        new_action = self.sim.data.qpos + action

        self.do_simulation(new_action, self.frame_skip)

        diff_vector = self.get_site_xpos("insert_site") - self.get_site_xpos("base_site")
        x_vector = diff_vector[0]
        y_vector = diff_vector[1]
        dist = np.linalg.norm(diff_vector)
        dist_x = np.linalg.norm(x_vector)
        dist_y = np.linalg.norm(y_vector)

        if dist_x < 0.005 and self.counter != 0:  # Millimiters
            done = False
            self.reward_x = 100
            if dist_y < 0.0006:
                done = True
                self.reward_y = 100
        else:
            done = False
            reward_done = 0

        # Force Reward
        f = self.sim.data.sensordata.flat[:]

        force_v = np.linalg.norm(f[:3])/5           # 5 max force, normalization
        torque_v = np.linalg.norm(f[3:6])

        self.force_sum += force_v
        self.reward_force = np.mean(self.force_sum)

        # print(f"reward_force: {self.reward_force}")

        reward_pos = 1 - (dist/self.dist_max)**0.2
        reward_xd = 1 - (dist_x/self.dist_max)**0.2
        reward_yd = 1 - (dist_y/self.dist_max)**0.2
        reward = self.reward_x + self.reward_y + (reward_xd + reward_yd)*0.5 - self.counter           # - self.reward_force/10

        # print(f"reward_pos: {reward_pos}")
        # print(f"reward_done: {reward_done}")
        # print(f"reward_force: {self.reward_force}")
        # print(f"reward: {reward}")

        self.counter += 1

        info = {}
        ob = self._get_obs()

        return ob, reward, done, info

    def _get_obs(self):
        # print(self.sim.data.qpos)
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:],
                self.sim.data.sensordata.flat[:],
                (self.get_site_xpos("insert_site") - self.get_site_xpos("base_site")).flat[:],
            ]
        ).astype(np.float32)

    def reset_model(self):
        qpos = np.asarray(self.init_qpos)
        c_xy = 0.05
        c_z = 0.01
        c_a = 0.1
        self.counter = 0
        self.force_sum = 0
        qpos[:2] = self.np_random.uniform(low=-c_xy, high=c_xy, size=2)
        qpos[2:3] = self.np_random.uniform(low=-c_z, high=c_z, size=1)
        qpos[3:6] = self.np_random.uniform(low=-c_a, high=c_a, size=3)
        # print(qpos[3:6])
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.4

    def get_site_xpos(self, site_name):
        return self.data.get_site_xpos(site_name)
