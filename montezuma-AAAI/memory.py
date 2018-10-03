import numpy as np
import random

from collections import namedtuple
Experience = namedtuple('Experience', 's g g_id a r tilde_r sp intrinsic_done terminal man_loc')

class ExperienceMemory():
	def __init__(self, size=100000):
		self.size = size
		self.memory = []

	def push(self,*experience):
		if self.memory.__len__() < self.size:
			self.memory.append(*experience)			
		else:
			self.memory.pop(0)
			self.memory.append(*experience)

	def sample_controller(self,batch_size=32):
		length = self.memory.__len__()
		self.indecies = []
		states = np.empty([batch_size,4,84,84], dtype=np.uint8)
		subgoals = np.empty([batch_size,1,84,84], dtype=np.uint8)
		actions = np.empty([batch_size], dtype=np.int32)
		rewards = np.empty([batch_size], dtype=np.float32)
		intrinsic_dones = np.empty([batch_size], dtype=np.uint8)
		state_primes = np.empty([batch_size,4,84,84], dtype=np.uint8)

		for i in range(0,batch_size):
			j = random.randint(0,length-1)
			e = self.memory[j]
			states[i,:,:,:] = e.s
			subgoals[i,:,:,:] = e.g
			actions[i] = e.a
			rewards[i] = e.tilde_r
			state_primes[i] = e.sp
			intrinsic_dones[i] = e.intrinsic_done 
			self.indecies.append(j)
		return states, subgoals, actions, rewards, state_primes, intrinsic_dones
	
	def sample_meta_controller(self):
		pass

	def get_man_positions(self):
		length = self.memory.__len__()
		X = np.empty([length,2])
		for i in range(0,length):
			X[i,:] = self.memory[i].man_loc
		return X

	def __len__(self):
		return len(self.memory)
