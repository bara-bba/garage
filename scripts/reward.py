import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy import genfromtxt

TEST = 10
MIN = 0.1
MAX = 1.1

dist_step = 0.0006
max_episode_length = 300

dist_max = dist_step * max_episode_length * np.sqrt(3)
dist = np.arange(0, dist_max, dist_step)

exp = np.arange(MIN, MAX, np.linalg.norm(MAX - MIN)/TEST)

for i in range(TEST):
    reward = -(dist/dist_max)**exp[i]
    # reward = np.interp(reward, (reward.min(), reward.max()), (-1, 1))
    plt.plot(dist, reward, label=f"exp = {exp[i]:.2f}")

plt.title("Exponential Reward Function")
plt.legend()
plt.xlim([0, 0.30])
plt.show()
