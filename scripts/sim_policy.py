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


def query_yes_no(question, default='yes'):
    """Ask a yes/no question via raw_input() and return their answer.

    Args:
        question (str): Printed to user.
        default (str or None): Default if user just hits enter.

    Raises:
        ValueError: If the provided default is invalid.

    Returns:
        bool: True for "yes"y answers, False for "no".

    """
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--max_episode_length',
                        type=int,
                        default=1000,
                        help='Max length of episode')
    args = parser.parse_args()

    with tf.compat.v1.Session() as sess:
        with open(args.file, 'rb') as pickle_file:
            data = cloudpickle.load(pickle_file)
            policy = data['algo'].policy
            env = normalize(GymEnv(PandaEnv()), normalize_reward=True, normalize_obs=True)
            while True:
                path = rollout(env,
                               policy,
                               max_episode_length=args.max_episode_length,
                               animated=True)
