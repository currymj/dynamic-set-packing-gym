# from pytorch examples repository
import argparse
from collections import defaultdict
import gym
import numpy as np
from itertools import count
import gym_dynamic_set_packing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--plot-name', type=str, default='rewards_plot', help='name to save rewards plot file')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--ep-length', type=int, default=20, help='max length of episode')
parser.add_argument('--num-episodes', type=int, default=1000, help='number of episodes')
parser.add_argument('--env-name', type=str, default='DynamicSetPacking-adversarial-v0',help='environment name')
args = parser.parse_args()


env = gym.make(args.env_name)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self, input_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        #self.affine1 = nn.Linear(input_size, 2, bias=False)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        #action_scores = self.affine1(x)
        return F.softmax(action_scores, dim=1)


policy = Policy(env.observation_space.shape[0])
#policy.affine1.data = torch.Tensor([[1.0,1.0,-1.0,-1.0,-1.0],[1.0,1.0,1.0,1.0,1.0]])
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    mean_rewards = []
    ep_rewards = []
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(args.ep_length):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        mean_rewards.append(running_reward)
        ep_rewards.append(ep_reward)
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            print('match probs on [1.0,1.0,1.0,1.0,1.0]: {}'.format(policy(torch.Tensor([[1.0,1.0,1.0,1.0,1.0]]))))
            print('match probs on [1.0,1.0,0.0,0.0,0.0]: {}'.format(policy(torch.Tensor([[1.0,1.0,0.0,0.0,0.0]]))))
            print('match probs on [0.0,0.0,0.0,0.0,0.0]: {}'.format(policy(torch.Tensor([[0.0,0.0,0.0,0.0,0.0]]))))
        if i_episode > args.num_episodes:
            break
    plt.plot(ep_rewards)
    plt.plot(mean_rewards)
    plt.savefig(args.plot_name)
    torch.save(policy.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
