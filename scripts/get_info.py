#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.spatial.transform import Rotation as R

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from panda_env_meta_base import PandaEnvMetaBase
from panda_env import PandaEnv
import matplotlib.pyplot as plt

import os
import csv

# Gaussian Distribution Noise UR5 Parameters
MUX = 0.175
SIX = 0.181
MUY = -0.118
SIY = 0.125
MUZ = -0.508
SIZ = 0.185

env = PandaEnv()

obs_dim = int(np.prod(env.observation_space.shape))
action_dim = int(np.prod(env.action_space.shape))

MAX = 30000

diff = np.zeros(MAX)
fx = np.zeros(MAX)
fy = np.zeros(MAX)
fz = np.zeros(MAX)
time = np.zeros(MAX)

t = 0

with open('panda_force.csv', 'w') as outfile:
    out = csv.writer(outfile)
    for i in range(MAX):

        env.sim.data.ctrl[0] = -0.10          # panda_x
        env.sim.data.ctrl[1] = -0.0           # panda_y
        env.sim.data.ctrl[2] = -0.0          # panda_z

        env.sim.data.ctrl[3] = 0          # panda_ball_1
        env.sim.data.ctrl[4] = 0          # panda_ball_2    --> IN QPOS WRITTEN AS QUATERNION!
        env.sim.data.ctrl[5] = 0          # panda_ball_3


        # # RADIANS
        # print("X" + str(env.sim.model.joint_name2id("insert_x")) + ": " + str(env.sim.data.qpos.flat[0]))
        # print("Y" + str(env.sim.model.joint_name2id("insert_y")) + ": " + str(env.sim.data.qpos.flat[1]))
        # print("Z" + str(env.sim.model.joint_name2id("insert_z")) + ": " + str(env.sim.data.qpos.flat[2]))
        # print("a" + str(env.sim.model.joint_name2id("insert_ball_1")) + ": " + str(env.sim.data.qpos.flat[3]))
        # print("b" + str(env.sim.model.joint_name2id("insert_ball_2")) + ": " + str(env.sim.data.qpos.flat[4]))
        # print("c" + str(env.sim.model.joint_name2id("insert_ball_3")) + ": " + str(env.sim.data.qpos.flat[5]))
        #
        # # print(np.shape(env.sim.data.qpos))
        #
        # # r = R.from_quat(env.sim.data.qpos[3:])
        # # state = np.hstack([env.sim.data.qpos[:3], r.as_euler('xyz', degrees=True)]) #CONVERSION BTW RELATIVE AND ABSOLUTE POS ROTATION MATRIX; SEE XML FRAMES!
        #
        # state = env.sim.data.qpos
        #
        # # print(env.sim.get_state())
        # print("STATE: " + str(state))
        #
        xpos_ee = env.sim.data.get_site_xpos("ee_site")
        xpos_base = env.sim.data.get_site_xpos("base_site")
        xpos_insert = env.sim.data.get_site_xpos("insert_site")

        diff_vector = xpos_insert - xpos_base
        # print(diff_vector)
        #
        distance = np.linalg.norm(diff_vector)
        #
        # print("BASE_SITE: " + str(xpos_base))
        # print("INSERT_SITE: " + str(xpos_insert))
        #
        # print("DISTANCE: " + str(distance))

        force_x = (env.sim.data.sensordata[0])
        force_y = (env.sim.data.sensordata[1])
        force_z = (env.sim.data.sensordata[2])
        # print(env.sim.data.sensordata.shape)

        force = env.sim.data.sensordata.flat[:]
        force[0] += np.random.normal(MUX, SIX)
        force[1] += np.random.normal(MUY, SIY)
        force[2] += np.random.normal(MUZ, SIZ)

        print(xpos_ee-xpos_insert)

        env.sim.step()
        t += 1
        env.render()


        # fx[t-1] = (force[0])
        # fy[t-1] = (force[1])
        # fz[t-1] = (force[2])
        # # f[t-1] = (force - 0.35)/5
        # diff[t-1] = np.linalg.norm(env.sim.data.ctrl[0] - env.sim.data.get_site_xpos("insert_site")[0])
        # time[t-1] = env.sim.get_state().time
        #
        # row = [fx[t-1], fy[t-1], fz[t-1]]
        # out.writerow(row)

# outfile.close()
#
#
# plt.plot()
# plt.show()