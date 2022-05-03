#!/usr/bin/env python3
"""Apply pre-learned policy to UR5."""

import argparse
import sys
from garage.envs import normalize

steps, max_steps = 0, 10

param_dir = "/home/bara/PycharmProjects/garage/data/local/experiment/garage_sac_panda_position/"

import cloudpickle
import tensorflow as tf

from garage.experiment import Snapshotter
snapshotter = Snapshotter()

from wrappers import LowPassFilterWrapper
from ur5_env_old import UR5Env
from panda_env import PandaEnv

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

    with tf.compat.v1.Session() as sess:
        data = snapshotter.load(param_dir)
    policy = data['algo'].policy
    env = (PandaEnv())

    from garage import rollout
    path = rollout(env, policy)
    print(path)