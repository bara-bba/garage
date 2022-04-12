#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
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

env = PandaEnv()

obs_dim = int(np.prod(env.observation_space.shape))
action_dim = int(np.prod(env.action_space.shape))

diff = np.zeros(1000)
time = np.zeros(1000)

t = 0

for i in range(1000):

    env.sim.data.ctrl[0] = 0.01          # panda_x
    env.sim.data.ctrl[1] = 0          # panda_y
    env.sim.data.ctrl[2] = 0         # panda_z

    env.sim.data.ctrl[3] = 0          # panda_ball_1
    env.sim.data.ctrl[4] = 0          # panda_ball_2    --> IN QPOS WRITTEN AS QUATERNION!
    env.sim.data.ctrl[5] = 0          # panda_ball_3


    # RADIANS
    print("X" + str(env.sim.model.joint_name2id("insert_x")) + ": " + str(env.sim.data.qpos.flat[0]))
    print("Y" + str(env.sim.model.joint_name2id("insert_y")) + ": " + str(env.sim.data.qpos.flat[1]))
    print("Z" + str(env.sim.model.joint_name2id("insert_z")) + ": " + str(env.sim.data.qpos.flat[2]))
    print("a" + str(env.sim.model.joint_name2id("insert_ball_1")) + ": " + str(env.sim.data.qpos.flat[3]))
    print("b" + str(env.sim.model.joint_name2id("insert_ball_2")) + ": " + str(env.sim.data.qpos.flat[4]))
    print("c" + str(env.sim.model.joint_name2id("insert_ball_3")) + ": " + str(env.sim.data.qpos.flat[5]))


    # print(np.shape(env.sim.data.qpos))

    # r = R.from_quat(env.sim.data.qpos[3:])
    # state = np.hstack([env.sim.data.qpos[:3], r.as_euler('xyz', degrees=True)]) #CONVERSION BTW RELATIVE AND ABSOLUTE POS ROTATION MATRIX; SEE XML FRAMES!

    state = env.sim.data.qpos

    # print(env.sim.get_state())
    print("STATE: " + str(state))

    xpos_base = env.sim.data.get_site_xpos("base_site")
    xpos_insert = env.sim.data.get_site_xpos("insert_site")

    distance = np.linalg.norm(xpos_insert - xpos_base)

    print("BASE_SITE: " + str(xpos_base))
    print("INSERT_SITE: " + str(xpos_insert))

    print("DISTANCE: " + str(distance))

    force = env.sim.data.qfrc_actuator + env.sim.data.qfrc_passive
    # force = env.sim.data.sensordata
    print("FORCE: " + str(force))


    # print("XPOS object" + str(env.sim.model.body_name2id("target")) + ": " + str(xpos))

    # print("NU: " + str(env.sim.model.nu))
    # print(type(env.sim.model.nu))

    # print("NQ: " + str(env.sim.model.nq))
    # print(type(env.sim.model.nq))

    # print("NV: " + str(env.sim.model.nv))
    # print(type(env.sim.model.nv))

    # print(str(env.act ion_space))

    env.sim.step()
    t += 1
    env.render()

    diff[t-1] = np.linalg.norm(env.sim.data.ctrl[0] - env.sim.data.get_site_xpos("insert_site")[0])
    time[t-1] = env.sim.get_state().time


#PRINT FORCES TOO!

plt.plot(time, diff)
plt.show()