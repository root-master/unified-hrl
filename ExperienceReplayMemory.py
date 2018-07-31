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
		new_X = np.array((s[0]/16,s[1]/16,r/100)).reshape(1,-1)
		if self.X.size == 0:
			self.X = new_X
		else:
			self.X = np.append(self.X,new_X,axis=0)





