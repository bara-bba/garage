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

    env.sim.data.ctrl[0] = 0         # panda_x
    env.sim.data.ctrl[1] = 0          # panda_y
    env.sim.data.ctrl[2] = 0         # panda_z

    env.sim.data.ctrl[3] = 0          # panda_ball_1
    env.sim.data.ctrl[4] = 0          # panda_ball_2    --> IN QPOS WRITTEN AS QUATERNION!
    env.sim.data.ctrl[5] = 0          # panda_ball_3

    state = env.sim.data.qpos

    # print(env.sim.get_state())
    print("STATE: " + str(state))

    xpos_base = env.sim.data.get_site_xpos("base_site")
    xpos_insert = env.sim.data.get_site_xpos("insert_site")
    xpos_tcp = env.sim.data.get_site_xpos("ee_site")

    distance = np.linalg.norm(xpos_insert - xpos_base)


    print("BASE_SITE: " + str(xpos_base-xpos_base))
    print("INSERT_SITE: " + str(xpos_insert-xpos_base))
    print("TCP_SITE: " + str(xpos_tcp-xpos_base))

    print(xpos_insert-xpos_tcp)

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