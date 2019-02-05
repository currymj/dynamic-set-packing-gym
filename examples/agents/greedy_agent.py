import gym
from gym import spaces
import gym_dynamic_set_packing

class GreedyMatchAgent:
    "A simple agent for the 0/1 problem that always matches."
    def __init__(self):
        self.action_space = spaces.Discrete(2)

    def act(self, observation, reward, done):
        return 1

if __name__ == '__main__':
    env = gym.make('DynamicSetPacking-silly-v0')
    agent = GreedyMatchAgent()

    episode_count = 10
    reward = 0.0
    done = False
    max_steps = 100

    for i in range(episode_count):
        ob = env.reset()
        while i in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print('reward: {}, new state: {}'.format(reward, env.render()))

env.close()
