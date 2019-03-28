import gym
from gym import spaces
import gym_dynamic_set_packing
import time

class SmartMatchAgent:
    "A simple agent for this weird adversarial system that always matches if the last type is available."
    def __init__(self):
        self.action_space = spaces.Discrete(2)

    def act(self, observation, reward, done):
        if observation[2] > 0:
            return 1
        else:
            return 0

if __name__ == '__main__':
    env = gym.make('DynamicSetPacking-adversarial-v0')
    agent = SmartMatchAgent()

    episode_count = 10
    reward = 0.0
    done = False
    max_steps = 100

    start_time = time.time()
    for i in range(episode_count):
        failure_count = 0
        print('episode {}'.format(i))
        ob = env.reset()
        total_reward = 0.0
        for i in range(max_steps):
            action = agent.act(ob, reward, done)
            found_third = ob[2] > 0
            ob, reward, done, _ = env.step(action)
            if found_third and (reward < 3):
                failure_count += 1
                print('failure!')

            total_reward += reward
            print('reward: {}, new state: {}'.format(reward, env.render()))
        print('total episode reward: {}, failures: {}'.format(total_reward, failure_count))
    end_time = time.time()
    print('time taken: {} s'.format(end_time - start_time))

env.close()
