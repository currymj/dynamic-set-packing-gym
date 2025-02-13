import gym
from gym import spaces
import gym_dynamic_set_packing
import time

class GreedyMatchAgent:
    "A simple agent for the 0/1 problem that always matches."
    def __init__(self):
        self.action_space = spaces.Discrete(2)

    def act(self, observation, reward, done):
        return 1

if __name__ == '__main__':
    env = gym.make('DynamicSetPacking-gurobitest-v0')
    agent = GreedyMatchAgent()

    episode_count = 10
    reward = 0.0
    done = False
    max_steps = 100

    start_time = time.time()
    for i in range(episode_count):
        print('episode {}'.format(i))
        ob = env.reset()
        total_reward = 0.0
        for i in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            print('reward: {}, new state: {}'.format(reward, env.render()))
        print('total episode reward: {}'.format(total_reward))
    end_time = time.time()
    print('time taken: {} s'.format(end_time - start_time))

env.close()
