import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

# Gaussian Distribution Noise UR5 Parameters
MUX = 0.175
SIX = 0.181
MUY = -0.118
SIY = 0.125
MUZ = -0.508
SIZ = 0.185


class PandaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.counter = 0
        self.reward_force = 0
        self.force_sum = 0
        self.dist_max = 0.0006*300*np.sqrt(3)
        mujoco_env.MujocoEnv.__init__(self, "/home/bara/PycharmProjects/garage/panda/insert_base.xml", 100)

        utils.EzPickle.__init__(self)

    def step(self, action):

        # print(f"Action: {action}")
        new_action = self.sim.data.qpos + action
        # print(f"NewAction: {new_action}")

        self.do_simulation(new_action, self.frame_skip)

        diff_vector = self.get_site_xpos("insert_site") - self.get_site_xpos("base_site")
        dist = np.linalg.norm(diff_vector)

        if dist < 0.004:  # Millimiters
            done = True
            # print("Done")
            reward_done = 1
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

        reward_pos = -(dist/self.dist_max)**0.2
        # reward_pos = -dist
        reward = reward_pos + reward_done           # - self.reward_force/10

        # print(f"reward_pos: {reward_pos}")
        # print(f"reward_done: {reward_done}")
        # print(f"reward_force: {self.reward_force}")
        # print(f"reward: {reward}")

        self.counter += 1

        info = {}
        ob = self._get_obs()

        # print(f"Obs: {ob}")
        # print(f"Rew: {reward}")

        return ob, reward, done, info

    def _get_obs(self):
        # print(self.sim.data.qpos)
        qpos = self.sim.data.qpos.flat[:]

        force = self.sim.data.sensordata.flat[:]
        # force[0] += np.random.normal(0, SIX)
        # force[1] += np.random.normal(0, SIY)
        # force[2] += np.random.normal(0, SIZ)

        diff_vector = (self.get_site_xpos("insert_site") - self.get_site_xpos("base_site")).flat[:]

        return np.concatenate(
            [
                qpos,
                force,
                diff_vector,
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
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.4

    def get_site_xpos(self, site_name):
        return self.data.get_site_xpos(site_name)
