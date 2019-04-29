# starting point was from pytorch examples repository
import argparse
from collections import defaultdict, namedtuple
import gym
import numpy as np
from itertools import count
import gym_dynamic_set_packing
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
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--ep-length', type=int, default=20, help='max length of episode')
parser.add_argument('--batches-per-ep', type=int, default=1, help='minibatches per episode')
args = parser.parse_args()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
env = gym.make('DynamicSetPacking-adversarial-v0')
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self, input_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        #self.affine1 = nn.Linear(input_size, 2, bias=False)

        self.saved_log_probs = defaultdict(list)
        self.rewards = defaultdict(list)
        self.saved_states = defaultdict(list)

    def save_log_prob(self, log_prob, batch_num):
        self.saved_log_probs[batch_num].append(log_prob)

    def save_state(self, state, batch_num):
        self.saved_states[batch_num].append(state)

    def save_reward(self, reward, batch_num):
        self.rewards[batch_num].append(reward)

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


def select_action(state, batch_num):
    state = torch.from_numpy(state).float().unsqueeze(0)
    policy.save_state(state, batch_num)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    #policy.saved_log_probs.append(m.log_prob(action))
    policy.save_log_prob(m.log_prob(action), batch_num)
    return action.item()


def finish_episode(batch_count):
    R = 0
    policy_loss = []

    disc_returns = {}
    # first compute discounted returns
    for b in range(batch_count):
        returns = []
        for r in policy.rewards[b][::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        disc_returns[b] = returns

    # then get mean and std of all combined
    all_rewards = torch.cat(tuple(disc_returns.values()))
    returns_mean = all_rewards.mean()
    returns_std = all_rewards.std()


    # then compute policy loss
    for b in range(batch_count):
        ret_b = disc_returns[b]
        ret_b = (ret_b - returns_mean) / (returns_std + eps)
        for log_prob, R in zip(policy.saved_log_probs[b], ret_b):
            policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    for k,v in policy.rewards.items():
        del v[:]
    for k, v in policy.saved_log_probs.items():
        del v[:]
    for k, v in policy.saved_states.items():
        del v[:]



def main():
    running_reward = 10
    for i_episode in count(1):
        for b in range(args.batches_per_ep):
            state, ep_reward = env.reset(), 0
            action_counts = defaultdict(int)
            for t in range(args.ep_length):  # Don't infinite loop while learning
                action = select_action(state, b)
                action_counts[action] += 1
                state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                #policy.rewards.append(reward)
                policy.save_reward(reward, b)
                ep_reward += reward
                if done:
                    break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(args.batches_per_ep)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            print('0: {}, 1: {}'.format(action_counts[0], action_counts[1]))
            print('match probs on [1.0,1.0,1.0,1.0,1.0]: {}'.format(policy(torch.Tensor([[1.0,1.0,1.0,1.0,1.0]]))))
            print('match probs on [1.0,1.0,0.0,0.0,0.0]: {}'.format(policy(torch.Tensor([[1.0,1.0,0.0,0.0,0.0]]))))
            print('match probs on [0.0,0.0,0.0,0.0,0.0]: {}'.format(policy(torch.Tensor([[0.0,0.0,0.0,0.0,0.0]]))))


if __name__ == '__main__':
    main()
