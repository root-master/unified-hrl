from model import DQN
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.transforms as T

class Controller():
	def __init__(self,
				 experience_memory=None,
				 lr = 0.00025,
				 alpha=0.95,
				 eps=0.01,
				 batch_size=32,
				 gamma=0.99,
				 load_pretrained=False,
				 saved_model_path='./models/a.model'):
		
		self.experience_memory = experience_memory # expereince replay memory
		self.lr = lr # learning rate
		self.alpha = alpha # optimizer parameter
		self.eps = 0.01 # optimizer parameter
		self.gamma = 0.99	
		# BUILD MODEL 
		USE_CUDA = torch.cuda.is_available()
		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")

		dfloat_cpu = torch.FloatTensor
		dfloat_gpu = torch.cuda.FloatTensor

		dlong_cpu = torch.LongTensor
		dlong_gpu = torch.cuda.LongTensor

		duint_cpu = torch.ByteTensor
		dunit_gpu = torch.cuda.ByteTensor 
		
		dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
		duinttype = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

		self.dtype = dtype
		self.dlongtype = dlongtype
		self.duinttype = duinttype

		Q = DQN(in_channels=5, num_actions=18).type(dtype)		
		if load_pretrained:			
			Q.load_state_dict(torch.load(saved_model_path))
		Q_t = DQN(in_channels=5, num_actions=18).type(dtype)
		Q_t.load_state_dict(Q.state_dict())
		Q_t.eval()
		for param in Q_t.parameters():
			param.requires_grad = False

		Q = Q.to(self.device)
		Q_t = Q_t.to(self.device)

		# if torch.cuda.device_count() > 0:
		# 	Q = nn.DataParallel(Q).to(self.device)
		# 	Q_t = nn.DataParallel(Q_t).to(self.device)
		# 	batch_size = batch_size * torch.cuda.device_count()
		# else:
		# 	batch_size = batch_size

		self.batch_size = batch_size
		self.Q = Q
		self.Q_t = Q_t
		# optimizer
		optimizer = optim.RMSprop(Q.parameters(),lr=lr, alpha=alpha, eps=eps)
		self.optimizer = optimizer
		print('init: Controller --> OK')

	def get_best_action(self,s,g):
		x = np.concatenate((s,g),axis=0).reshape((1,5,84,84))
		q = self.Q.forward(torch.Tensor(x).type(self.dtype)/255.0)
		q_np = q.cpu().detach().numpy()
		return q_np.argmax()

	def compute_Q(self,s,g):
		x = np.concatenate((s,g),axis=0).reshape((1,5,84,84))
		q = self.Q.forward(torch.Tensor(x).type(self.dtype)/255.0)
		q_np = q.cpu().detach().numpy()
		return q_np

	def update_w(self):
		states, subgoals, actions, rewards, state_primes, intrinsic_dones = \
						self.experience_memory.sample_controller(batch_size=self.batch_size)
		x = np.concatenate((states,subgoals),axis=1)
		x = torch.Tensor(x)	
		xp = np.concatenate((state_primes,subgoals),axis=1)
		xp = torch.Tensor(xp)
		actions = torch.Tensor(actions).type(self.dlongtype)
		rewards = torch.Tensor(rewards).type(self.dtype)
		intrinsic_dones = torch.Tensor(intrinsic_dones).type(self.dtype)
		# sending data to gpu
		dtype = self.dtype
		if torch.cuda.is_available():
			with torch.cuda.device(0):
				x = torch.Tensor(x).to(self.device).type(dtype)/255.0
				xp = torch.Tensor(xp).to(self.device).type(dtype)/255.0
				actions = actions.to(self.device)
				rewards = rewards.to(self.device)
				intrinsic_dones = intrinsic_dones.to(self.device)
		# forward path
		q = self.Q.forward(x)
		q = q.gather(1, actions.unsqueeze(1))
		q = q.squeeze()
		
		q_p1 = self.Q.forward(xp)
		_, a_prime = q_p1.max(1)

		q_t_p1 = self.Q_t.forward(xp)
		q_t_p1 = q_t_p1.gather(1, a_prime.unsqueeze(1))
		q_t_p1 = q_t_p1.squeeze()

		error = rewards + self.gamma * (1 - intrinsic_dones) * q_t_p1 - q
		target = rewards + self.gamma * (1 - intrinsic_dones) * q_t_p1
		clipped_error = -1.0 * error.clamp(-1, 1)
		
		self.optimizer.zero_grad()
		q.backward(clipped_error.data.unsqueeze(1))
		
		# We can use Huber loss for smoothness
		# loss = F.smooth_l1_loss(q, target)
		# loss.backward()
		
		for param in self.Q.parameters():
			param.grad.data.clamp_(-1, 1)
		
		# update weights
		self.optimizer.step()

	def update_target_params(self):
		self.Q_t.load_state_dict(self.Q.state_dict())

	def save_model(self, model_save_path):
		torch.save(self.Q.state_dict(), model_save_path)


class MetaController():
	def __init__(self,				 
				 meta_controller_experience_memory=None,
				 lr = 0.00025,
				 alpha=0.95,
				 eps=0.01,
				 batch_size=32,
				 gamma=0.99,
				 num_options=12):
		# expereince replay memory
		self.meta_controller_experience_memory = meta_controller_experience_memory
		self.lr = lr # learning rate
		self.alpha = alpha # optimizer parameter
		self.eps = 0.01 # optimizer parameter
		self.gamma = 0.99	
		# BUILD MODEL 
		USE_CUDA = torch.cuda.is_available()
		if torch.cuda.is_available() and torch.cuda.device_count()>1:
			self.device = torch.device("cuda:1")
		elif torch.cuda.device_count()==1:
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")

		dfloat_cpu = torch.FloatTensor
		dfloat_gpu = torch.cuda.FloatTensor

		dlong_cpu = torch.LongTensor
		dlong_gpu = torch.cuda.LongTensor

		duint_cpu = torch.ByteTensor
		dunit_gpu = torch.cuda.ByteTensor 
		
		dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
		duinttype = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

		self.dtype = dtype
		self.dlongtype = dlongtype
		self.duinttype = duinttype

		Q = DQN(in_channels=4, num_actions=num_options).type(dtype)		
		Q_t = DQN(in_channels=4, num_actions=num_options).type(dtype)
		Q_t.load_state_dict(Q.state_dict())
		Q_t.eval()
		for param in Q_t.parameters():
			param.requires_grad = False

		Q = Q.to(self.device)
		Q_t = Q_t.to(self.device)

		self.batch_size = batch_size
		self.Q = Q
		self.Q_t = Q_t
		# optimizer
		optimizer = optim.RMSprop(Q.parameters(),lr=lr, alpha=alpha, eps=eps)
		self.optimizer = optimizer
		print('init: Meta Controller --> OK')
		
	def get_best_option(self,s):
		x = s.reshape((1,4,84,84))
		x = torch.Tensor(x).type(self.dtype).to(self.device)
		q = self.Q.forward(x/255.0)
		q_np = q.cpu().detach().numpy()
		return q_np.argmax()

	def compute_Q(self,s):
		x = s.reshape((1,4,84,84))
		x = torch.Tensor(x).type(self.dtype).to(self.device)
		q = self.Q.forward(x/255.0)
		q_np = q.cpu().detach().numpy()
		return q_np

	def update_w(self):
		if self.meta_controller_experience_memory.__len__() < 2:
			return
		states0, subgoal_ids, returns, state_primes0, terminals = \
			self.meta_controller_experience_memory.sample_meta_controller(batch_size=self.batch_size)
		x = states0
		x = torch.Tensor(x)	
		xp = state_primes0
		xp = torch.Tensor(xp)
		subgoal_ids = torch.Tensor(subgoal_ids).type(self.dlongtype)
		returns = torch.Tensor(returns).type(self.dtype)
		terminals = torch.Tensor(terminals).type(self.dtype)
		# sending data to gpu
		dtype = self.dtype

		x = torch.Tensor(x).to(self.device).type(dtype)/255.0
		xp = torch.Tensor(xp).to(self.device).type(dtype)/255.0
		subgoal_ids = subgoal_ids.to(self.device)
		returns = returns.to(self.device)
		terminals = terminals.to(self.device)

		# forward path
		q = self.Q.forward(x)
		q = q.gather(1, subgoal_ids.unsqueeze(1))
		q = q.squeeze()
		
		q_p1 = self.Q.forward(xp)
		_, g_prime = q_p1.max(1)

		q_t_p1 = self.Q_t.forward(xp)
		q_t_p1 = q_t_p1.gather(1, g_prime.unsqueeze(1))
		q_t_p1 = q_t_p1.squeeze()

		error = returns + self.gamma * (1 - terminals) * q_t_p1 - q
		target = returns + self.gamma * (1 - terminals) * q_t_p1
		clipped_error = -1.0 * error.clamp(-1, 1)
		
		self.optimizer.zero_grad()
		q.backward(clipped_error.data)
		
		for param in self.Q.parameters():
			param.grad.data.clamp_(-1, 1)
		
		# update weights
		self.optimizer.step()

	def update_target_params(self):
		self.Q_t.load_state_dict(self.Q.state_dict())

	def save_model(self, model_save_path):
		torch.save(self.Q.state_dict(), model_save_path)




		
