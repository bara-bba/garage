#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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

from panda_env import PandaEnv

"""Snapshotter snapshots training data.

When training, it saves data to binary files. When resuming,
it loads from saved data.

Args:
    snapshot_dir (str): Path to save the log and iteration snapshot.
    snapshot_mode (str): Mode to save the snapshot. Can be either "all"
        (all iterations will be saved), "last" (only the last iteration
        will be saved), "gap" (every snapshot_gap iterations are saved),
        "gap_and_last" (save the last iteration as 'params.pkl' and save
        every snapshot_gap iteration separately), "gap_overwrite" (same as
        gap but overwrites the last saved snapshot), or "none" (do not
        save snapshots).
    snapshot_gap (int): Gap between snapshot iterations. Wait this number
        of iterations before taking another snapshot.

"""

@wrap_experiment(snapshot_mode='last')
def garage_sac_panda_position(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)

    trainer = Trainer(snapshot_config=ctxt)
    env = normalize(GymEnv(PandaEnv(), max_episode_length=1000))

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=100,
              max_episode_length_eval=1000,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              reward_scale=1.,
              steps_per_epoch=1)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=3000, batch_size=1000, plot=True)


s = np.random.randint(0, 1000)
garage_sac_panda_position(seed=521)