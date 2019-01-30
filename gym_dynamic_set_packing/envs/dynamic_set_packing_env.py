import gym
from gym import spaces
import numpy as np
import random

class DynamicSetPackingEnv(gym.Env):

    def __init__(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        raise NotImplementedError

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
