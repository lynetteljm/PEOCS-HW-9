# my packages
import models
import utils

# python packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import os

class Base(object):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
		p_lr: float, v_lr: float):

		# set device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# initalize actor & critic
		self.actor = models.FCNetwork(input_dim=input_dim, hidden_dim=hidden_dim, 
			output_dim=output_dim).to(self.device)	
		self.critic = models.FCNetwork(input_dim=input_dim, hidden_dim=hidden_dim,
			output_dim=1).to(self.device)

		# initalize optimizers
		self.optim_actor = optim.Adam(self.actor.parameters(), lr=p_lr)
		self.optim_critic = optim.Adam(self.critic.parameters(), lr=v_lr)


	def act(self, state):
		# transform state into tensor
		t_state = torch.tensor(state)\
			.reshape(1, -1)\
			.float().to(self.device)

		# run policy
		with torch.no_grad():
			logits = self.actor(t_state)

		# sample from action distribution
		actions_dist = Categorical(logits=logits)
		action = actions_dist.sample().item()
		return action


	def save(self, path: str):
		path_actor = os.path.join(path, "actor.pth")
		torch.save(self.actor.state_dict(), path_actor)
		path_critic = os.path.join(path, "critic.pth")
		torch.save(self.critic.state_dict(), path_critic)