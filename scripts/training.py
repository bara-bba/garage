#!/usr/bin/env python3
"""Load and Training using existing Q-functions in SAC (Simulation)"""
import numpy as np
import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, Snapshotter
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler, DefaultWorker
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy, DeterministicMLPPolicy
from garage.trainer import Trainer, TFTrainer
from garage.torch.q_functions import ContinuousMLPQFunction
from torch.nn import functional as F

from panda_env import PandaEnv
# env = normalize(GymEnv(PandaEnv(), max_episode_length=300), normalize_reward=False, normalize_obs=False)
from ur5_env import UR5Env
env = normalize(GymEnv(UR5Env(), max_episode_length=300), normalize_reward=False, normalize_obs=True)

@wrap_experiment(snapshot_mode='last')
def training(ctxt=None, seed=1):

    deterministic.set_seed(seed)

    trainer = Trainer(snapshot_config=ctxt)

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[64, 64],
        hidden_nonlinearity=None,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    snapshotter = Snapshotter()
    snapshot = snapshotter.load("/home/bara/PycharmProjects/garage/data/local/experiment/sac_2_normalized")

    qf1 = snapshot['algo']._qf1
    qf2 = snapshot['algo']._qf2

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=DefaultWorker,
                           n_workers=1)

    sac = SAC(env_spec=env.spec,
              policy=policy,
              policy_lr=0.001,
              qf1=qf1,
              qf2=qf2,
              qf_lr=0.001,
              sampler=sampler,
              optimizer=torch.optim.Adam,
              gradient_steps_per_itr=100,
              replay_buffer=replay_buffer,
              min_buffer_size=int(1e1),
              num_evaluation_episodes=1,
              discount=0.99,
              buffer_batch_size=128,
              initial_log_entropy=0.,
              reward_scale=1.,
              steps_per_epoch=1
              # eval_env=normalize(GymEnv(PandaEnv(), max_episode_length=300), normalize_reward=True, normalize_obs=True)
              )

    set_gpu_mode(False)
    sac.to()
    print("Training...")
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=100, batch_size=64, plot=False)


s = np.random.randint(0, 1000)
training(seed=s)
