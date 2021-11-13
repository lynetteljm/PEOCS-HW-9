import torch

class Buffer(object):
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.states  = []
		self.actions = []
		self.rewards = []
		self.done 	 = []

	def append(self, state, action, reward, done):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.done.append(done)

	def clear(self):
		del self.states[:]
		del self.actions[:]
		del self.rewards[:]
		del self.done[:]

	def get_states(self):
		return torch.tensor(self.states)\
			.float().to(self.device)

	def get_actions(self):
		return torch.tensor(self.actions)\
			.reshape(-1, 1)\
			.long().to(self.device)

	def get_rewards(self):
		return torch.tensor(self.rewards)\
			.float().to(self.device)

	def get_done(self):
		return torch.tensor(self.done)\
			.float().to(device)

	def get_rewards2go(self, gamma=0.99, normalize=True):
		discounted = 0
		rewards2go = [] 

		for r, done in reversed(list(zip(self.rewards, self.done))):
			if done:
				discounted = 0 
			discounted = r + gamma * discounted
			rewards2go = [discounted] + rewards2go

		rewards2go = torch.tensor(rewards2go)
		if normalize:
			eps = 1e-4
			rewards2go = (rewards2go - rewards2go.mean()) / (rewards2go.std() + eps)
		
		return rewards2go.to(self.device)