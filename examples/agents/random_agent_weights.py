import gym
from gym import spaces
import gym_dynamic_set_packing
import numpy as np

class RandomMatchAgent:
    "A simple agent for the type weight problem that spits out type weights uniform on [-1,1]."
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def act(self, observation, reward, done):
        return 2.0*np.random.rand(self.action_dim) - 1.0

if __name__ == '__main__':
    env = gym.make('DynamicSetPacking-weightedtest-v0')
    print(env.action_space.shape[0])
    agent = RandomMatchAgent(env.action_space.shape[0])

    episode_count = 10
    reward = 0.0
    done = False
    max_steps = 100

    for i in range(episode_count):
        print('episode {}'.format(i))
        ob = env.reset()
        total_reward = 0.0
        for i in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            print('action taken: {}, reward: {}, new state: {}'.format(action, reward, env.render()))
        print('total episode reward: {}'.format(total_reward))

env.close()
