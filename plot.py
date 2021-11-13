import pandas as pd
import numpy as np
import os

from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="LunarLander-v2")
parser.add_argument("--algos", nargs="+", type=str, default=["TRPO"])
parser.add_argument("--steps_int", type=int, default=1000)
parser.add_argument("--n_steps", type=int, default=1000000)
args = parser.parse_args()


def uniform_sample(x, y, x0, ):
	flinear = interpolate.interp1d(x, y)
	x_new = np.arange(x0, args.n_steps, args.steps_int)
	y_new = flinear(x_new)
	return x_new, y_new

def plot_algo(env_name: str, algo: str, data):
	path = os.path.join("ckpts", env_name, algo)
	dirs = os.listdir(path)
	
	logs = []
	step0 = 0

	for seed in dirs:
		log_path = os.path.join(path, seed, "logs.csv")
		log = pd.read_csv(log_path)
		logs.append(log)
		step0 = max(step0, log["step"][0])

	for log in logs:
		step = log["step"].to_numpy()
		score = log["score"].to_numpy()
		step, score = uniform_sample(step, score, step0)

		data["steps"]  += list(step)
		data["scores"] += list(score)
		data["algo"]   += [algo] * len(step)
		


if __name__ == "__main__":
	data = { "steps": [], "scores": [], "algo": []}

	for algo in args.algos:
		plot_algo(args.env_name, algo, data)

	df = pd.DataFrame.from_dict(data)
	sns.lineplot(x=data["steps"], y=data["scores"], hue=data["algo"], ci="sd")
	plt.xlabel("step")
	plt.ylabel("score")
	plt.title(args.env_name)
	plt.show()
