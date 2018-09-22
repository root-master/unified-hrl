import numpy as np
import random

from collections import namedtuple
Experience = namedtuple('Experience', 's g g_id a r r_tilde sp intrinsic_done terminal man_loc')
ExperienceMeta = namedtuple('ExperienceMeta', 's0 g_id G s0p terminal')

class ExperienceMemory():
	def __init__(self, size=1000000):
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
		actions = np.empty([batch_size], dtype=np.uint8)
		rewards = np.empty([batch_size], dtype=np.float32)
		intrinsic_dones = np.empty([batch_size], dtype=np.uint8)
		state_primes = np.empty([batch_size,4,84,84], dtype=np.uint8)

		for i in range(0,batch_size):
			j = random.randint(0,length-1)
			e = self.memory[j]
			states[i,:,:,:] = e.s
			subgoals[i,:,:,:] = e.g
			actions[i] = e.a
			rewards[i] = e.r_tilde
			state_primes[i] = e.sp
			intrinsic_dones[i] = e.intrinsic_done 
			self.indecies.append(j)
		return states, subgoals, actions, rewards, state_primes, intrinsic_dones
	
	def get_man_positions(self):
		length = self.memory.__len__()
		X = np.empty([length,2])
		for i in range(0,length):
			X[i,:] = self.memory[i].man_loc
		return X

	def __len__(self):
		return len(self.memory)

class ExperienceMemoryMeta():
	def __init__(self, size=50000):
		self.size = size
		self.memory = []

	def push(self,*experience):
		if self.memory.__len__() < self.size:
			self.memory.append(*experience)			
		else:
			self.memory.pop(0)
			self.memory.append(*experience)

	def sample_meta_controller(self,batch_size=32):
		length = self.memory.__len__()
		batch_size = min([batch_size,self.__len__()])
		self.indecies = []
		states0 = np.empty([batch_size,4,84,84], dtype=np.uint8)
		subgoal_ids = np.empty([batch_size], dtype=np.uint8)
		returns = np.empty([batch_size], dtype=np.float32)
		terminals = np.empty([batch_size], dtype=np.uint8)
		state_primes0 = np.empty([batch_size,4,84,84], dtype=np.uint8)

		for i in range(0,batch_size):
			j = random.randint(0,length-1)
			e = self.memory[j]
			states0[i,:,:,:] = e.s0
			subgoal_ids[i] = e.g_id
			returns[i] = e.G
			state_primes0[i] = e.s0p
			terminals[i] = e.terminal 
			self.indecies.append(j)
		return states0, subgoal_ids, returns, state_primes0, terminals
	
	def __len__(self):
		return len(self.memory)
