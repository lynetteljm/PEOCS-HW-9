import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
		super(FCNetwork, self).__init__()

		# save input paramters
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		# create layers
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3=  nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = self.fc3(x)
		return x