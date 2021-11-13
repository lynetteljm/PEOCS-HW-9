from .base import *
from torch import autograd
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence


class TRPO(Base):
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
		p_lr: float=7e-4, v_lr: float=1e-4, damping_coeff: float=0.1, cg_iters: int=10, 
		backtrack_iters: int=10, backtrack_coeff: float=0.5, delta=0.01):
		
		super(TRPO, self).__init__(
			input_dim=input_dim, 
			hidden_dim=hidden_dim, 
			output_dim=output_dim,
			p_lr=p_lr,
			v_lr=v_lr
		)

		# create online actor
		self.old_actor = models.FCNetwork(input_dim=input_dim, hidden_dim=hidden_dim, 
			output_dim=output_dim).to(self.device)
		self.old_actor.load_state_dict(self.actor.state_dict())

		# additional parameters to trpo
		self.damping_coeff 	 = damping_coeff
		self.cg_iters 		 = cg_iters
		self.backtrack_iters = backtrack_iters
		self.backtrack_coeff = backtrack_coeff
		self.delta 			 = delta


	def update(self, buff):
		# get data from buffer
		s  = buff.get_states() 	
		a  = buff.get_actions()
		r  = buff.get_rewards()
		rg = buff.get_rewards2go()

		# pass state through actor and 
		# select the appropriate action
		logits   = self.actor(s)
		log_dist = F.log_softmax(logits, dim=1)
		log_prob = torch.gather(log_dist, 1, a).squeeze(1)
		dist = Categorical(logits=logits)

		old_logits = self.old_actor(s)
		old_dist = Categorical(logits=old_logits)

		# pass state throgh critic
		values = self.critic(s).squeeze(1)

		# compute jacobian
		self.actor.zero_grad()
		baseline = values.detach()
		actor_loss = log_prob * (rg - baseline) 
		g = self.jacobian(actor_loss.mean(), self.actor).reshape(-1, 1)
		
		# compute hessian
		self.actor.zero_grad()
		kl_loss = kl_divergence(dist, old_dist).mean()
		grad = autograd.grad(kl_loss, self.actor.parameters(), create_graph=True)
		H = self.hessian(grad, self.actor)
		H += self.damping_coeff * np.eye(H.shape[0]) 

		# solve H^(-1) g by conjugate gradient
		x = np.zeros_like(g)
		x = self.conjugate_gradient(H, g, x).reshape(-1, 1)

		# compute alpha by backtrack line search
		alpha = self.backtrack(x, H, old_dist, s)

		# update weights
		self.update_weights(alpha, x, H)
		
		# update old actor
		self.old_actor.load_state_dict(self.actor.state_dict())

		# optimization step for critic
		critic_loss = (values - rg)**2
		self.optim_critic.zero_grad()
		critic_loss.mean().backward()
		self.optim_critic.step()


	def update_weights(self, alpha, x, H):
		# upate weights
		grads = np.sqrt(2 * self.delta / np.matmul(x.T, np.matmul(H, x))) * x
		grad_weights = self.vec2weights(grads.reshape(-1))

		i = 0
		for param in self.actor.parameters():
			param.data = param.data + alpha * grad_weights[i]
			i += 1

	def jacobian(self, loss, model):
		grad = autograd.grad(loss, model.parameters(), create_graph=True)
		np_grads = []

		for g in grad:
			np_grads.append(g.reshape(-1).cpu().data.numpy())
		return np.concatenate(np_grads)

	def hessian(self, loss_grad, model):
		cnt = 0
		for g in loss_grad:
			g_vector = g.contiguous().view(-1) if cnt == 0 \
				else torch.cat([g_vector, g.contiguous().view(-1)])
			cnt = 1
		
		l = g_vector.size(0)
		hessian = torch.zeros(l, l)
		for idx in range(l):
			grad2rd = autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
			cnt = 0

			for g in grad2rd:
				g2 = g.contiguous().view(-1) if cnt == 0\
					else torch.cat([g2, g.contiguous().view(-1)])
				cnt = 1
			hessian[idx] = g2

		return hessian.cpu().data.numpy()

	def vec2weights(self, g):
		cnt = 0
		weights = []

		for param in self.actor.parameters():
			rows, cols = param.shape[0], 0
			
			if len(param.shape) > 1:
				cols = param.shape[1]

			n_cells = rows * cols if cols > 0 else rows
			vec = g[cnt: cnt + n_cells]
			vec = torch.tensor(vec.reshape(rows, cols) if cols > 0 else vec.reshape(-1)).float().to(self.device)
			weights.append(vec)
			cnt += n_cells

		return weights

	def conjugate_gradient(self, A, b, x):
		r = b - np.dot(A, x)
		p = r
		rsold = np.dot(np.transpose(r), r)

		for i in range(self.cg_iters):
			Ap = np.dot(A, p)
			alpha = (rsold / np.dot(np.transpose(p), Ap)).item()
			x = x + np.dot(alpha, p)
			r = r - np.dot(alpha, Ap)
			rsnew = np.dot(np.transpose(r), r)
			if np.sqrt(rsnew) < 1e-8:
				break
			p = r + (rsnew/rsold)*p
			rsold = rsnew

		return x

	def backtrack(self, x, H, old_dist, s):
		alpha = 1.0

		for i in range(self.backtrack_iters):
			alpha *= self.backtrack_coeff
			self.update_weights(alpha, x, H)

			# compute output distribution for the new weights
			logits  = self.actor(s)
			dist = Categorical(logits=logits)

			# compute the mean KL divergence
			mean_kl = kl_divergence(dist, old_dist).mean()
			if mean_kl < self.delta:
				break

			# restore the change
			self.actor.load_state_dict(self.old_actor.state_dict())

		return alpha