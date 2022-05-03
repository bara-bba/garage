from garage.experiment import Snapshotter
from panda_env import PandaEnv

snapshotter = Snapshotter()
snapshot = snapshotter.load('/home/bara/PycharmProjects/garage/data/local/experiment/garage_sac_panda_position')

from ur5_env_old import UR5Env

# Setup new experiment
from garage import wrap_experiment
from garage.sampler import LocalSampler
from garage.torch.algos import BC
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer
import tensorflow as tf



@wrap_experiment
def bc_with_pretrained_expert(ctxt=None):
    trainer = Trainer(ctxt)
    policy = GaussianMLPPolicy(env.spec, [8, 8])
    batch_size = 1000
    sampler = LocalSampler(agents=expert,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=expert,
              sampler=sampler,
              policy_lr=1e-2,
              loss='log_prob')
    trainer.setup(algo, env)
    trainer.train(10, batch_size=batch_size)

with tf.compat.v1.Session() as sess:
    expert = snapshot['algo'].policy
    # env = snapshot['env']  # We assume env is the same
    env = UR5Env()
    bc_with_pretrained_expert()
