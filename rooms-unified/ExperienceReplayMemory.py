import numpy as np
class ExperienceReplayMemory():
	def __init__(self,memory_size=400):
		self.memory = []
		self.memory_size = memory_size
		self.X = np.array([[]])

	def push(self, *experience):
		if len( self.memory) < self.memory_size:
			self.push_X(*experience)
			self.memory.append(*experience)			
		else:
			self.X = np.delete(self.X, obj=0, axis=0)
			self.push_X(*experience)
			self.memory.pop(0)
			self.memory.append(*experience)
	
	def push_X(self, experience):
		s = experience[0]
		a = experience[1]
		r = experience[2]
		sp = experience[3]
		new_X = np.array((s[0]/16,s[1]/16)).reshape(1,-1)
		if self.X.size == 0:
			self.X = new_X
		else:
			self.X = np.append(self.X,new_X,axis=0)

	def pop(self):
		self.memory.pop()
		self.X = np.delete(self.X, obj=-1, axis=0)

	def get_reward_np(self):
		return np.array(self.get_rewards()).reshape(-1, 1)

	def get_rewards(self):
		rewards = []
		for i in range(len(self.memory)):
			rewards.append(self.memory[i][2])
		return rewards

	def get_experience_X_np(self):
		self.X = np.array([[]])
		for (i, experience) in enumerate(self.memory):
			s = experience[0]
			a = experience[1]
			r = experience[2]
			sp = experience[3]
			new_X = np.array((s[0]/16,s[1]/16,r/40,sp[0]/16,sp[1]/16)).reshape(1,-1)
			if i == 0:
				self.X = new_X
			else:
				self.X = np.append(self.X,new_X,axis=0)






