from __future__ import print_function

import argparse
import os
import pickle

import gym
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import tf_util
from load_policy import load_policy

metadata = {
    'Ant-v2': (111, 8),
    'HalfCheetah-v2': (17, 6),
    'Hopper-v2': (11, 3),
    'Humanoid-v2': (376, 17),
    'Reacher-v2': (11, 2),
    'Walker2d-v2': (17, 6)
}


class Rollout(Dataset):
    def __init__(self, exp_name):
        assert exp_name in metadata
        self.act, self.obs = load_data(exp_name)  # expert_data
        if len(self.act.shape) == 3:
            self.act = np.squeeze(self.act, 1)
        if len(self.obs.shape) == 3:
            self.obs = np.squeeze(self.obs, 1)

    def aggregate(self, act, obs):
        if len(act.shape) == 3:
            act = np.squeeze(act, 1)
        if len(obs.shape) == 3:
            obs = np.squeeze(obs, 1)
        self.act = np.concatenate([self.act, act])
        self.obs = np.concatenate([self.obs, obs])

    def __getitem__(self, index):
        return self.obs[index], self.act[index]

    def __len__(self):
        return len(self.act)


class Policy(nn.Module):
    def __init__(self, act_dim, obs_dim):
        super(Policy, self).__init__()
        self.fc0 = nn.Linear(act_dim, 128)
        self.fc1 = nn.Linear(128, obs_dim)

    def forward(self, x):
        x = x.type_as(self.fc0.bias)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x  # Continuous control problem


def load_data(exp_name):
    file_name = os.path.join("expert_data", exp_name)
    with open(file_name + ".pkl", "rb") as f:
        data = pickle.load(f)
    act = data["actions"]
    obs = data["observations"]
    print("File {} loaded! The action space {}, the observation space {}".format(file_name, act.shape, obs.shape))
    return act, obs


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def run_policy(env, policy, device, episode):
    policy.eval()
    rewards = []
    total_rewards = []
    observations = []
    actions = []
    with torch.no_grad():
        for _ in range(episode):
            state = env.reset()
            done = False
            total_reward = 0.
            while not done:
                observations.append(state)
                action = policy(torch.from_numpy(state).to(device)).cpu().detach().numpy()
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                actions.append(action)
                total_reward += reward
            total_rewards.append(total_reward)
    return actions, observations, rewards, total_rewards


def aggregate(args, env, policy, expert_policy, dataset, device):
    _, obs, _, _ = run_policy(env, policy, device, args.dagger)
    obs = np.stack(obs)
    act = expert_policy(obs)
    dataset.aggregate(act, obs)
    print("Successfully aggregated data. New dataset has shapes: act {}, obs {}.".format(dataset.act.shape,
                                                                                         dataset.obs.shape))
    return None


def test(args, env, policy, device):
    _, _, _, rewards = run_policy(env, policy, device, args.rollouts)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,default='HalfCheetah-v2')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--rollouts', type=int, default=20, help="The number of episode ran at test period")
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dagger', type=int, default=-1, metavar='N',
                        help='number of episodes in collecting DAgger data (default: -1)')
    parser.add_argument('--iteration', type=int, default=10, metavar='N',
                        help='running iteration for DAgger. (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # setup pytorch
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # setup dataset
    dataset = Rollout(args.env)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # setup policy and expert
    policy = Policy(*metadata[args.env]).to(device)
    expert_policy = load_policy("expert/" + args.env + ".pkl")

    # setup training
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    critertion = nn.MSELoss()

    # setup RL environment
    env = gym.make(args.env)
    env.seed(args.seed)

    means = []
    stds = []

    # the TF session is used for running expert.
    with tf.Session():
        tf.set_random_seed(args.seed)
        tf_util.initialize()
        # for BC, args.iter=1.
        for iteration in range(args.iteration):
            for epoch in range(args.epochs):
                train(args, policy, critertion, device, train_loader, optimizer, epoch)
            rewards = test(args, env, policy, device)

            print(
                "Iteration {} end. Test of env {}. Run for {} rollouts. Mean Rewards {}. Standard Deviation {}.".format(
                    iteration, args.env, args.rollouts,
                    np.mean(rewards),
                    np.std(rewards)))

            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
            # for BC dagger defaulr=-1
            if args.dagger > 0:
                aggregate(args, env, policy, expert_policy, dataset, device)

    print("Mean", means)
    print("Std", stds)


if __name__ == '__main__':
    main()
