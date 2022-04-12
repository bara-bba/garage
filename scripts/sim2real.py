import tensorflow as tf
from garage.experiment import Snapshotter

import rtde_control
import rtde_receive

from wrappers import LowPassFilterWrapper

from ur5_env import UR5Env

param_dir = "/home/bara/PycharmProjects/Garage/data/local/experiment/garage_sac_panda_position/"
snapshotter = Snapshotter()

with tf.compat.v1.Session() as sess:
    print("extrating parameters from file %s ..." % param_dir)
    data = snapshotter.load(param_dir)

policy = data['algo'].policy
# env = LowPassFilterWrapper(UR5Env())
env = UR5Env()

from garage import rollout

path = rollout(env, policy)
print(path)
