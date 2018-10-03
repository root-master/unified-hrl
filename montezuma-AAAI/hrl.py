from model import DQN
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

class MetaLearning():
	def __init__(self):
		# build models
		self.Qt = DQN(in_channels=5, num_actions=18) # Controller Q network
		self.Qt_t = DQN(in_channels=5, num_actions=18) # Controller target network
		# self.meta_controller = Model(in_channels=4, num_actions=10)
		self.Q = None # Meta-Controller Q network
		self.Q_t = None # Meta-Controller target network






