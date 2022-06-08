#!/usr/bin/env python3
"""Simulates pre-learned policy."""
import argparse
import sys

import cloudpickle
import tensorflow as tf
import gym

from garage import rollout
from panda_env import PandaEnv
from garage.envs import GymEnv, normalize


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--max_episode_length',
                        type=int,
                        default=1000,
                        help='Max length of episode')
    args = parser.parse_args()

    print(args.file)

    with open(args.file, 'rb') as pickle_file:
        data = cloudpickle.load(pickle_file)
        policy = data['algo'].policy
        env = data['env']
        # env = normalize(GymEnv(PandaEnv(), max_episode_length=300), normalize_obs=True, normalize_reward=False)
        while True:
            path = rollout(env,
                           policy,
                           max_episode_length=args.max_episode_length,
                           animated=True)
