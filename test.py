import algos

import torch
import gym
import argparse
import os
from PIL import Image
from itertools import count
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, help="name of the environment",default="LunarLander-v2")
parser.add_argument("--algo", type=str, help="algorithm used: [REINFROCE, PPO, TRPO]", default="TRPO")
parser.add_argument("--hidden_dim", type=int, help="hidden dimension of the network", default=32)
parser.add_argument("--seed", type=int, help="seed to be tested (need to have a model trained with this seed)",default=10)
parser.add_argument("--n_episodes", type=int, help="# of episodes to test the policy", default=10)
args = parser.parse_args()


def process_images(imgs: list):
	for i in range(len(imgs)):
		imgs[i] = Image.fromarray(imgs[i])
		w, h = imgs[i].size
		imgs[i] = imgs[i].resize((w//2, h//2))

	# create folder
	if not os.path.exists("GIFs"):
		os.makedirs("GIFs")

	path = os.path.join("GIFs", "%s_%s_%s.gif" % (args.env_name, args.algo, str(args.seed)))
	imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def test():
	# create environment
	env = gym.make(args.env_name)
	input_dim = env.observation_space.shape[0]
	output_dim = env.action_space.n

	# create algorith object
	if args.algo == "REINFORCE":
		algo = algos.REINFORCE(
			input_dim=input_dim,
			hidden_dim=args.hidden_dim, 
			output_dim=output_dim, 
		)
	elif args.algo == "PPO":
		algo = algos.PPO(
			input_dim=input_dim,
			hidden_dim=args.hidden_dim,
			output_dim=output_dim,
		)
	elif args.algo == "TRPO":
		algo = algos.TRPO(
			input_dim=input_dim,
			hidden_dim=args.hidden_dim, 
			output_dim=output_dim, 
		)
	else:
		raise ValueError("Incorrect algorithm!"
		"The values available are: REINFORCE, PPO, TRPO")
	
	# load model
	path = os.path.join("ckpts", args.env_name, args.algo, str(args.seed), "actor.pth")
	algo.actor.load_state_dict(torch.load(path))

	# run for some episodes
	episode_returns = []
	imgs = {}

	for episode in range(args.n_episodes):
		imgs[episode] = []
		state = env.reset()
		sum_rewards = 0

		for i in count():
			# get action
			action = algo.act(state)

			# get next_state and reward
			next_state, reward, done, info = env.step(action=action)
			state = next_state
			sum_rewards += reward

			# save image
			imgs[episode].append(env.render(mode='rgb_array'))

			if done:
				break

		episode_returns.append(sum_rewards)
		print("episode %d, return: %.2f" % (episode, sum_rewards))

	# close environment
	env.close()

	# process images
	idx = np.argmax(episode_returns)
	process_images(imgs[idx])

	# final statstics
	mean = np.mean(episode_returns)
	std = np.std(episode_returns)
	min = np.min(episode_returns)
	max = np.max(episode_returns)
	median = np.median(episode_returns)

	print("=" * 50)
	print("# episodes: %d" % (args.n_episodes))
	print("mean: %.2f, std: %.2f" % (mean, std))
	print("median: %.2f, min: %.2f, max: %.2f" % (median, min, max))
	print("=" * 50)


if __name__ == "__main__":
	test()
