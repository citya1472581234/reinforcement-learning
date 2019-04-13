"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Soroush Nasiriany, Sid Reddy, and Greg Kahn
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


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(GaussianPolicy, self).__init__()
        self.mean = Net(obs_dim, act_dim)
        self.std = nn.Parameter(torch.ones(1, act_dim))

    def forward(self, obs):
        mean = torch.tanh(self.mean(obs))
        dist = torch.distributions.Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        return action, log_prob, torch.zeros((log_prob.size(0), 1))


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CategoricalPolicy, self).__init__()
        self.network = Net(obs_dim, act_dim)

    def forward(self, obs):
        logit = self.network(obs)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1)


def normalize(array):  # Normalize Numpy array or PyTorch tensor to a standard normal distribution.
    return (array - array.mean()) / (array.std() + 1e-7)


def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getfullargspec(train_AC)[0]
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
    logz.save_agent(agent)


def make_agent_args(args, env):
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    return {'n_layers': args['n_layers'],
            'ob_dim': env.observation_space.shape[0],
            'ac_dim': env.action_space.n if discrete else env.action_space.shape[0],
            'discrete': discrete,
            'size': args['size'],
            'learning_rate': args['learning_rate'],
            'num_target_updates': args['num_target_updates'],
            'num_grad_steps_per_target_update': args['num_grad_steps_per_target_update'],
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'animate': args['render'],
            'max_path_length': args['ep_len'] if args['ep_len'] > 0 else env.spec.max_episode_steps,
            'min_timesteps_per_batch': args['batch_size'],
            'gamma': args['discount'],
            'normalize_advantages': not args['dont_normalize_advantages']}


# ============================================================================================#
# Actor Critic
# ============================================================================================#

class Agent(object):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.n_layers = args['n_layers']
        self.ob_dim = args['ob_dim']
        self.ac_dim = args['ac_dim']
        self.discrete = args['discrete']
        self.size = args['size']
        self.learning_rate = args['learning_rate']
        self.num_target_updates = args['num_target_updates']
        self.num_grad_steps_per_target_update = args['num_grad_steps_per_target_update']
        self.device = args['device']
        self.animate = args['animate']
        self.max_path_length = args['max_path_length']
        self.min_timesteps_per_batch = args['min_timesteps_per_batch']
        self.gamma = args['gamma']
        self.normalize_advantages = args['normalize_advantages']

        self.actor = None  # The loss function is computed at self.update_parameters.
        if self.discrete:
            self.actor = CategoricalPolicy(self.ob_dim, self.ac_dim).to(self.device)
        else:
            self.actor = GaussianPolicy(self.ob_dim, self.ac_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic_prediction = Net(self.ob_dim, 1).to(self.device)
        self.critic_loss = nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic_prediction.parameters(), lr=self.learning_rate)

    def sample_trajectories(self, itr, env):
        # Collect trajectories until we have enough timesteps
        timesteps_this_batch = 0
        obs, acs, next_obs, log_probs, res, terminals = [], [], [], [], [], []
        while timesteps_this_batch <= self.min_timesteps_per_batch:
            animate_this_episode = (len(res) == 0 and (itr % 10 == 0) and self.animate)
            ob, ac, next_ob, log_prob, re, terminal = self.sample_trajectory(env, animate_this_episode)
            obs.append(ob)
            acs.append(ac)
            next_obs.append(next_ob)
            log_probs.append(log_prob)
            res.append(re)
            terminals.append(terminal)
            timesteps_this_batch += len(re)
        return np.concatenate(obs), acs, np.concatenate(next_obs), torch.cat(log_probs).squeeze(1), res, np.concatenate(
            terminals), \
               timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, next_obs, log_probs, rewards, terminals = [], [], [], [], [], []
        for steps in count():
            if animate_this_episode:
                env.render()
                time.sleep(0.1)

            ac, log_prob, _ = self.actor(torch.from_numpy(ob).to(self.device).unsqueeze(0))
            if self.discrete:
                ac = ac[0].item()
            else:
                ac = ac[0].detach().cpu().numpy()
            obs.append(ob)
            acs.append(ac)
            log_probs.append(log_prob)
            next_ob, rew, done, _ = env.step(ac)
            next_obs.append(next_ob)
            rewards.append(rew)
            ob = next_ob
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
            # use terminals to count timestep end
        return np.array(obs, dtype=np.float32), \
               np.array(acs, dtype=np.float32), \
               np.array(next_obs, dtype=np.float32), \
               torch.cat(log_probs), \
               np.array(rewards, dtype=np.float32), \
               np.array(terminals, dtype=np.float32)

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        re_n = torch.from_numpy(np.concatenate(re_n)).to(self.device)
        ob_no = torch.from_numpy(ob_no).to(self.device)
        next_ob_no = torch.from_numpy(next_ob_no).to(self.device)
        mask = torch.from_numpy(1 - terminal_n).to(self.device)

        v_prime_n = self.critic_prediction(next_ob_no).reshape(re_n.shape)
        q_n = re_n + self.gamma * v_prime_n * mask
        v_n = self.critic_prediction(ob_no).reshape(re_n.shape)
        adv_n = q_n - v_n

        if self.normalize_advantages:
            adv_n = normalize(adv_n)
        return adv_n.detach()

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        # Use a bootstrapped target values to update the critic
        # Compute the target values r(s, a) + gamma*V(s') by calling the critic to compute V(s')
        # In total, take n=self.num_grad_steps_per_target_update*self.num_target_updates gradient update steps
        # Every self.num_grad_steps_per_target_update steps, recompute the target values
        # by evaluating V(s') on the updated critic

        ob_no = torch.from_numpy(ob_no).to(self.device).unsqueeze(0)
        next_ob_no = torch.from_numpy(next_ob_no).to(self.device).unsqueeze(0)
        re_n = torch.from_numpy(np.concatenate(re_n)).to(self.device)
        mask = torch.from_numpy(1 - terminal_n).to(self.device)

        for _ in range(self.num_target_updates):
            self.critic_prediction.eval()

            v_prime_n = self.critic_prediction(next_ob_no).reshape(re_n.shape)
            target = re_n + self.gamma * v_prime_n * mask
            target = target.detach()

            self.critic_prediction.train()

            for _ in range(self.num_grad_steps_per_target_update):
                prediction = self.critic_prediction(ob_no).reshape(re_n.shape)
                loss = self.critic_loss(input=prediction, target=target)
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
        self.critic_prediction.eval()

    def update_actor(self, ob_no, log_prob_na, adv_n):
        """ 
            Update the parameters of the policy.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        self.actor.train()
        self.optimizer.zero_grad()
        loss = -torch.mean(log_prob_na * adv_n)
        loss.backward()
        self.optimizer.step()
        self.actor.eval()


def train_AC(args, logdir, seed):
    start = time.time()
    setup_logger(logdir, locals())

    # Make the gym environment
    env = gym.make(args['env_name'])
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    env.seed(args['seed'])
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    agent_args = make_agent_args(args, env)
    agent = Agent(agent_args)  # estimate_return_args

    total_timesteps = 0
    for itr in range(args['n_iter']):
        print("********** Iteration %i ************" % itr)

        ob_no, _, next_ob_no, log_prob_na, re_n, terminal_n, timesteps_this_batch = agent.sample_trajectories(itr, env)

        # (1) update the critic, by calling agent.update_critic
        # (2) use the updated critic to compute the advantage by, calling agent.estimate_advantage
        # (3) use the estimated advantage values to update the actor, by calling agent.update_actor

        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        agent.update_actor(ob_no, log_prob_na, adv_n)

        # Log diagnostics
        total_timesteps += timesteps_this_batch
        quick_log(agent=agent, rewards=re_n, iteration=itr, start_time=start, timesteps_this_batch=timesteps_this_batch,
                  total_timesteps=total_timesteps)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='ac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=100)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    args.exp_name = 'ac_' + args.exp_name

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    processes = []
    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)
        p = Process(target=train_AC, args=(vars(args), os.path.join(logdir, '%d' % seed), seed))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
