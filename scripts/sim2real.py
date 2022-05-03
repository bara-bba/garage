import tensorflow as tf
from garage.experiment import Snapshotter

from wrappers import LowPassFilterWrapper
from garage.envs import normalize, GymEnv

from ur5_env import UR5Env
from panda_env import PandaEnv

param_dir = "/home/bara/PycharmProjects/garage/data/local/experiment/garage_sac_panda_position/"
snapshotter = Snapshotter()

with tf.compat.v1.Session() as sess:
    print("extrating parameters from file %s ..." % param_dir)
    data = snapshotter.load(param_dir)

policy = data['algo'].policy
# env = normalize(LowPassFilterWrapper(UR5Env()))
env = normalize(GymEnv(UR5Env()))
from garage import rollout

path = rollout(env, policy)
print(path)


