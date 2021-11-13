# my packages
import models
import utils
import algos

# python packages
import torch
import numpy as np
import random
import argparse
from itertools import count
import gym
import os
import pandas as pd


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, help="algorithm: [REINFORCE, TRPO, PPO]", default="TRPO")
parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30])
parser.add_argument("--env_name", type=str, help="envrionment's name", default="LunarLander-v2")
parser.add_argument("--horizon", type=int, help="maximum number of steps per environment", default=512)
parser.add_argument("--hidden_dim", type=int, help="number of nodes in hidden layer", default=32)
parser.add_argument("--p_lr", type=float, help="policy learning rate", default=7e-4)
parser.add_argument("--v_lr", type=float, help="value function learning rate", default=1e-4)
parser.add_argument("--n_epochs", type=int, help="PPO # epochs", default=4)
parser.add_argument("--eps_clip", type=float, help="PPO eps clip", default=0.1)
parser.add_argument("--entropy_coeff", type=float, help="PPO entropy coeff", default=0.01)
parser.add_argument("--damping_coeff", type=float, help="TRPO damping coef for hessian maxtir", default=0.1)
parser.add_argument("--cg_iters", type=int, help="TRPO # of iteration for conjugate gradient method", default=10) 
parser.add_argument("--backtrack_iters", type=int, help="TRPO # of iteraction for backtrack search", default=10) 
parser.add_argument("--backtrack_coeff", type=float, help="TRPO # alpha coeff in backtrack search", default=0.5)
parser.add_argument("--delta", type=float, help="TRPO KL divergence constraint", default=0.01) 
parser.add_argument("--n_steps", type=int, default=1000000)
parser.add_argument("--update_step", type=int, default=2048)
parser.add_argument("--log_int", type=int, default=100)
args = parser.parse_args()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(seed: int, seed_dir: str):
	# create environment
	env = gym.make(args.env_name)
	input_dim = env.observation_space.shape[0]
	output_dim = env.action_space.n

	print("-" * 50)
	print("### Algorithm %s, seed: %d ###" % (args.algo, seed))
	print("-" * 50) 
	
	# set seeds
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	# create buffer
	buff = utils.Buffer()

	# create algorith object
	if args.algo == "REINFORCE":
		algo = algos.REINFORCE(
			input_dim=input_dim,
			hidden_dim=args.hidden_dim, 
			output_dim=output_dim, 
			p_lr=args.p_lr,
			v_lr=args.v_lr
		)
	elif args.algo == "PPO":
		algo = algos.PPO(
			input_dim=input_dim,
			hidden_dim=args.hidden_dim,
			output_dim=output_dim,
			p_lr=args.p_lr,
			v_lr=args.v_lr,
			n_epochs=args.n_epochs,
			eps_clip=args.eps_clip,
			entropy_coeff=args.entropy_coeff
		)
	elif args.algo == "TRPO":
		algo = algos.TRPO(
			input_dim=input_dim,
			hidden_dim=args.hidden_dim, 
			output_dim=output_dim, 
			p_lr=args.p_lr,
			v_lr=args.v_lr,
			damping_coeff=args.damping_coeff, 
			cg_iters=args.cg_iters, 
			backtrack_iters=args.backtrack_iters, 
			backtrack_coeff=args.backtrack_coeff, 
			delta=args.delta
		)
	else:
		raise ValueError("Incorrect algorithm!"
		"The values available are: REINFORCE, PPO, TRPO")
			

	tstep = 0			# number of steps in the environment
	sum_rewards = 0		# total rewards accumulated
	logs = {"step": [], "score": []}

	for episode in count(1):
		state = env.reset()

		for i in range(1, args.horizon + 1):
			tstep += 1  # update number of steps

			# get action
			action = algo.act(state)

			# get next_state and reward
			next_state, reward, done, info = env.step(action=action)
			buff.append(state, action, reward, done)
			state = next_state
			sum_rewards += reward

			# update policy
			if tstep % args.update_step == 0:
				algo.update(buff)		# update policy
				buff.clear()			# clear history

			if done:
				break

		if episode % args.log_int == 0:
			avg_rewards = sum_rewards / args.log_int
			sum_rewards = 0
			logs["step"].append(tstep)
			logs["score"].append(avg_rewards)
			print("step: %d, score: %.2f" % (tstep, avg_rewards))

			# save models
			algo.save(seed_dir)

			# save logs
			df_logs = pd.DataFrame.from_dict(logs)
			logs_path = os.path.join(seed_dir, "logs.csv")
			df_logs.to_csv(logs_path, index=False)

			# stopping criterion
			if tstep > args.n_steps:
				return

if __name__ == "__main__":
	# create ckpt dir
	if not os.path.exists("ckpts"):
		os.makedirs("ckpts")

	# create environment dir
	env_dir = os.path.join("ckpts", args.env_name)
	if not os.path.exists(env_dir):
		os.makedirs(env_dir)

	# create algorithm dir
	algo_path = os.path.join(env_dir, args.algo)
	if not os.path.exists(algo_path):
		os.makedirs(algo_path)


	for seed in args.seeds:
		# create seed dir
		seed_dir = os.path.join(algo_path, str(seed))
		if not os.path.exists(seed_dir):
			os.makedirs(seed_dir)

		# train policy with the current seed
		train(seed, seed_dir)

	
