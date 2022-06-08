from garage.experiment import Snapshotter
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import time

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, Snapshotter
from garage import rollout

REAL = 1
MAX = 1000

step, max_episode_length = 0, 300

if REAL == 0:

     from panda_env import PandaEnv
     dir = "sim"

     snapshotter = Snapshotter()
     data = snapshotter.load('/home/bara/PycharmProjects/garage/data/local/experiment/sac_2_normalized')
     policy = data['algo'].policy
     env = normalize(GymEnv(PandaEnv(), max_episode_length=max_episode_length))

elif REAL == 1:

     # from ur5_env import UR5Env
     from panda_env import PandaEnv
     dir = "ur5"

     # env = normalize(UR5Env())
     env = normalize(GymEnv(PandaEnv(), max_episode_length=max_episode_length))

for i in range(MAX):

     print(f"{i/MAX*100:.1f}% \r")

     if REAL == 0:

          path = rollout(env, policy, animated=False)

          data = {"a[0]": np.transpose(path["actions"])[0],
                  "a[1]": np.transpose(path["actions"])[1],
                  "a[2]": np.transpose(path["actions"])[2],
                  "a[3]": np.transpose(path["actions"])[3],
                  "a[4]": np.transpose(path["actions"])[4],
                  "a[5]": np.transpose(path["actions"])[5],
                  "o[0]": np.transpose(path["observations"])[0],
                  "o[1]": np.transpose(path["observations"])[1],
                  "o[2]": np.transpose(path["observations"])[2],
                  "o[3]": np.transpose(path["observations"])[3],
                  "o[4]": np.transpose(path["observations"])[4],
                  "o[5]": np.transpose(path["observations"])[5],
                  "o[6]": np.transpose(path["observations"])[6],
                  "o[7]": np.transpose(path["observations"])[7],
                  "o[8]": np.transpose(path["observations"])[8],
                  "o[9]": np.transpose(path["observations"])[9],
                  "o[10]": np.transpose(path["observations"])[10],
                  "o[11]": np.transpose(path["observations"])[11],
                  "o[12]": np.transpose(path["observations"])[12],
                  "o[13]": np.transpose(path["observations"])[13],
                  "o[14]": np.transpose(path["observations"])[14],
                  "done": np.transpose(path["dones"])}


     elif REAL == 1:

            csv_file = (f"/home/bara/PycharmProjects/garage/sim2real_tf/sim/policy{i}.csv")
            df = pd.read_csv(csv_file)
            done = False
            obs = env.reset()
            step = 0

            a = np.empty(6)
            o = np.empty(15)
            d = np.empty(1)

            while step < max_episode_length and not done:

                 action = np.array(df.iloc[step][1:7])
                 env_step = env.step(action*6e-4)
                 obs, rew, done, _ = env_step.observation, env_step.reward, df["done"][step], {}

                 a = np.vstack((a, action))
                 o = np.vstack((o, obs))
                 d = np.vstack((d, done))

                 step += 1

            a = np.delete(a, (0), axis=0)
            o = np.delete(o, (0), axis=0)
            d = np.delete(d, (0), axis=0)

            data = {"a[0]": np.transpose(a)[0],
                    "a[1]": np.transpose(a)[1],
                    "a[2]": np.transpose(a)[2],
                    "a[3]": np.transpose(a)[3],
                    "a[4]": np.transpose(a)[4],
                    "a[5]": np.transpose(a)[5],
                    "o[0]": np.transpose(o)[0],
                    "o[1]": np.transpose(o)[1],
                    "o[2]": np.transpose(o)[2],
                    "o[3]": np.transpose(o)[3],
                    "o[4]": np.transpose(o)[4],
                    "o[5]": np.transpose(o)[5],
                    "o[6]": np.transpose(o)[6],
                    "o[7]": np.transpose(o)[7],
                    "o[8]": np.transpose(o)[8],
                    "o[9]": np.transpose(o)[9],
                    "o[10]": np.transpose(o)[10],
                    "o[11]": np.transpose(o)[11],
                    "o[12]": np.transpose(o)[12],
                    "o[13]": np.transpose(o)[13],
                    "o[14]": np.transpose(o)[14],
                    "done": np.transpose(d)[0]}


     df = pd.DataFrame(data=data)
     df.to_csv(f"/home/bara/PycharmProjects/garage/sim2real_tf/{dir}/policy{i}.csv")


