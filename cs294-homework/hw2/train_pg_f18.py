"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
"""
import inspect
import os
import time
from itertools import count

import gym
import numpy as np
import torch
from torch import nn
from torch.multiprocessing import Process
from torch.nn import functional as F

import logz


class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(obs_dim, 128)
        self.fc1 = nn.Linear(128, act_dim)

    def forward(self, x):
        x = x.type_as(self.fc0.bias)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        return x


class GaussianPolicy(nn.Module):   # continus action 
    def __init__(self, obs_dim, act_dim):
        super(GaussianPolicy, self).__init__()
        self.mean = Net(obs_dim, act_dim)
        self.std = nn.Parameter(torch.ones(1, act_dim))

    def forward(self, obs):
        mean = torch.tanh(self.mean(obs))   # normalize
        dist = torch.distributions.Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True) #log probability for loss BP
        return action, log_prob, torch.zeros((log_prob.size(0), 1))


class CategoricalPolicy(nn.Module):# discrete action 
    def __init__(self, obs_dim, act_dim):
        super(CategoricalPolicy, self).__init__()
        self.network = Net(obs_dim, act_dim)

    def forward(self, obs):
        logit = self.network(obs)
        dist = torch.distributions.Categorical(logits=logit)  # Bernoulli distribution  ,every action 1 or 0 probability
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1)


def normalize(array):  # Normalize Numpy array or PyTorch tensor to a standard normal distribution.
    return (array - array.mean()) / (array.std() + 1e-7)


def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getfullargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


def quick_log(agent, rewards, iteration, start_time, timesteps_this_batch, total_timesteps):
    returns = [re.sum() for re in rewards]
    ep_lengths = [len(re) for re in rewards]
    logz.log_dict({"Time": time.time() - start_time,
                   "Iteration": iteration,
                   "AverageReturn": np.mean(returns),
                   "StdReturn": np.std(returns),
                   "MaxReturn": np.max(returns),
                   "MinReturn": np.min(returns),
                   "EpLenMean": np.mean(ep_lengths),
                   "EpLenStd": np.std(ep_lengths),
                   "TimestepsThisBatch": timesteps_this_batch,
                   "TimestepsSoFar": total_timesteps})
    logz.dump_tabular()
    # logz.pickle_tf_vars()
    logz.save_agent(agent)

def make_agent_args(args, env):
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    return {'n_layers': args['n_layers'],
            'ob_dim': env.observation_space.shape[0],
            'ac_dim': env.action_space.n if discrete else env.action_space.shape[0],
            'discrete': discrete,
            'size': args['size'],
            'learning_rate': args['learning_rate'],
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'animate': args['render'],
            'max_path_length': args['ep_len'] if args['ep_len'] > 0 else env.spec.max_episode_steps,
            'min_timesteps_per_batch': args['batch_size'],
            'gamma': args['discount'],
            'reward_to_go': args['reward_to_go'],
            'nn_baseline': args['nn_baseline'],
            'normalize_advantages': not args['dont_normalize_advantages']}


class Agent(object):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.ob_dim = args['ob_dim']
        self.ac_dim = args['ac_dim']
        self.discrete = args['discrete']
        self.size = args['size']
        self.n_layers = args['n_layers']
        self.learning_rate = args['learning_rate']
        self.device = args['device']
        self.animate = args['animate']
        self.max_path_length = args['max_path_length']
        self.min_timesteps_per_batch = args['min_timesteps_per_batch']
        self.gamma = args['gamma']
        self.reward_to_go = args['reward_to_go']
        self.nn_baseline = args['nn_baseline']
        self.normalize_advantages = args['normalize_advantages']

        self.policy = None  # The loss function is computed at self.update_parameters.
        if self.discrete:
            self.policy = CategoricalPolicy(self.ob_dim, self.ac_dim).to(self.device)
        else:
            self.policy = GaussianPolicy(self.ob_dim, self.ac_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        if self.nn_baseline:
            self.baseline_prediction = Net(self.ob_dim, 1).to(self.device)  
            #value function  not reward
            self.baseline_loss = nn.MSELoss()
            self.baseline_optimizer = torch.optim.Adam(self.baseline_prediction.parameters(), lr=self.learning_rate)

    def sample_trajectories(self, itr, env):
        # Collect trajectories until we have enough timesteps
        timesteps_this_batch = 0
        obs, acs, res, log_probs = [], [], [], []
        while timesteps_this_batch <= self.min_timesteps_per_batch:
            animate_this_episode = (len(res) == 0 and (itr % 10 == 0) and self.animate)
            ob, ac, log_prob, re = self.sample_trajectory(env, animate_this_episode)
            obs.append(ob)
            acs.append(ac)
            res.append(re)
            log_probs.append(log_prob)
            timesteps_this_batch += len(re)
        return np.concatenate(obs), acs, torch.cat(log_probs), res, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards, log_probs = [], [], [], []
        for steps in count():
            if animate_this_episode: # show or not
                env.render()
                time.sleep(0.1)
            ac, log_prob, _ = self.policy(torch.from_numpy(ob).to(self.device).unsqueeze(0))
            if self.discrete:
                ac = ac[0].item()
            else:
                ac = ac[0].cpu().detach().numpy() # avoid BP to non-discrete part
            obs.append(ob)
            log_probs.append(log_prob.squeeze(-1))
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            if done or steps > self.max_path_length:
                break
        return np.array(obs), np.array(acs), torch.cat(log_probs), np.array(rewards),

    def sum_of_rewards(self, re_n):
        q_n = []
        for tau in re_n:
            result = []
            Q = 0
            for r in tau[::-1]:
                Q = r + self.gamma * Q   #gamma discont
                result.insert(0, Q)      # reward is not related before t'<t 
            if not self.reward_to_go:
                result = [Q for _ in range(len(tau))]  # reward is related t'<t  
            q_n.extend(result)
        return np.array(q_n)

    def compute_advantage(self, ob_no, q_n):
        q_n = torch.from_numpy(q_n).to(self.device)
        if self.nn_baseline:
            # b_n ???  reward ???
            b_n = self.baseline_prediction(torch.from_numpy(ob_no).to(self.device)) 
            # Baseline with shape [sum_of_path_lengths]
            b_n = normalize(b_n) * q_n.std() + q_n.mean()  
            # match the statistics of Q values
            adv_n = q_n - b_n.type_as(q_n)
        else:
            adv_n = q_n
        return adv_n

    def estimate_return(self, ob_no, re_n):
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_advantages:
            adv_n = normalize(adv_n)
        return q_n, adv_n

    def update_parameters(self, ob_no, log_prob_na, q_n, adv_n):
        if self.nn_baseline:
            self.baseline_prediction.train()
            prediction = self.baseline_prediction(torch.from_numpy(ob_no).to(self.device).unsqueeze(0))
            self.baseline_optimizer.zero_grad()
            target = normalize(torch.from_numpy(q_n).to(self.device)).type_as(prediction)
            loss = self.baseline_loss(input=prediction, target=target)
            loss.backward()
            self.baseline_optimizer.step()
            self.baseline_prediction.eval()
        self.policy.train()
        self.optimizer.zero_grad()
        loss = -torch.mean(log_prob_na.type_as(adv_n) * adv_n)  #minimize -J(),maximize J()
        loss.backward()
        self.optimizer.step()
        self.policy.eval()


def train_PG(args, logdir, seed):
    # Setup logger, random seed, gym environment, arguments, and agent.
    start = time.time()
    setup_logger(logdir, locals())

    env = gym.make(args['env_name'])
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    env.seed(args['seed'])
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    agent_args = make_agent_args(args, env)
    agent = Agent(agent_args)

    total_timesteps = 0
    for itr in range(args['n_iter']):
        print("********** Iteration %i ************" % itr)
        ob_no, _, log_prob_na, re_n, timesteps_this_batch = agent.sample_trajectories(itr, env)
        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        agent.update_parameters(ob_no, log_prob_na, q_n, adv_n)

        # Log diagnostics
        total_timesteps += timesteps_this_batch
        quick_log(agent=agent, rewards=re_n, iteration=itr, start_time=start, timesteps_this_batch=timesteps_this_batch,
                  total_timesteps=total_timesteps)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str,default='Walker2d-v2')
    parser.add_argument('--exp_name', type=str, default='rtg_only')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    processes = []
    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)
        p = Process(target=train_PG, args=(vars(args), os.path.join(logdir, '%d' % seed), seed))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
