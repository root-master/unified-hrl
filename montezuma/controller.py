from model import DQN
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Controller():
	def __init__(self,
				 lr = 0.00025,
				 alpha=0.95,
				 eps=0.01,
				 batch_size=32):
		
		# BUILD MODEL 
		USE_CUDA = torch.cuda.is_available()
		if torch.cuda.is_available():
			device0 = torch.device("cuda:0")
		else:
			device0 = torch.device("cpu")

		dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
		duinttype = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

		Qt = DQN(in_channels=5, num_actions=18).type(dtype)
		Qt_t = DQN(in_channels=5, num_actions=18).type(dtype)
		Qt_t.load_state_dict(Qt.state_dict())
		Qt_t.eval()
		for param in Qt_t.parameters():
			param.requires_grad = False

		if torch.cuda.device_count() > 0:
			Qt = nn.DataParallel(Qt).to(device0)
			Qt_t = nn.DataParallel(Qt_t).to(device0)
			batch_size = batch_size * torch.cuda.device_count()
		else:
			batch_size = batch_size

		
		# optimizer
		optimizer = optim.RMSprop(Qt.parameters(),lr=lr, alpha=alpha, eps=eps)

		
