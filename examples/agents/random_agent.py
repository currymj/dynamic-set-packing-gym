import gym
from gym import spaces
import gym_dynamic_set_packing
import numpy as np

class RandomMatchAgent:
    "A simple agent for the 0/1 problem that always matches."
    def __init__(self, match_prob):
        self.match_prob = match_prob
        self.action_space = spaces.Discrete(2)

    def act(self, observation, reward, done):
        if np.random.rand() <= self.match_prob:
            return 1
        else:
            return 0

if __name__ == '__main__':
    env = gym.make('DynamicSetPacking-silly-v0')
    agent = RandomMatchAgent(0.3)

    episode_count = 10
    reward = 0.0
    done = False
    max_steps = 100

    for i in range(episode_count):
        ob = env.reset()
        for i in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print('action taken: {}, reward: {}, new state: {}'.format(action, reward, env.render()))

env.close()
