#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py hw1/experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle

import gym
import load_policy
import numpy as np
import tensorflow as tf
import tf_util


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str,default='expert/Walker2d-v2.pkl')
    parser.add_argument('--envname', type=str,default='Walker2d-v2')
    parser.add_argument('--render', action='store_true',default=True) # show 
    parser.add_argument("--max_timesteps", type=int,default=1000)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        env.seed(1)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])   #observation input policy_fn
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r  # reward
                steps += 1
                if args.render:
                    env.render() # update env , and show 
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', args.envname + ('-test.pkl' if args.test else '.pkl')), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
