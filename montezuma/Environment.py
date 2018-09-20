import gym
import random
import numpy as np
class Environment():
	def __init__(self,
				task='MontezumaRevenge-v0',
				noop_max=30,
				skip_frame=4,
				repeat_action=4):
		self.task = task
		self.env = gym.make(task)
		self.noop_max = noop_max
		self.skip_frame = skip_frame
		self.repeat_action = repeat_action
		self.action_space = self.env.action_space

	def lives(self):
		return self.env.unwrapped.ale.lives()

	def reset(self):	
		s = self.env.reset()
		noop_random = random.randint(1, self.noop_max)
		for _ in range(noop_random):
			s,_,_,_ = self.env.step(0)
		S,_, _, _ = self.step(0)
		return S

	def skip_frames(self,a):	
		frames = []
		rewards = []
		for _ in range(self.skip_frame):
			s,r,done,step_info = self.env.step(a)
			frames.append(s)
			rewards.append(r)
		s_max = np.max(np.stack(frames), axis=0)
		total_rewards = sum(rewards)
		return s_max, total_rewards, done, step_info 

	def step(self,a):
		frames = []
		rewards = []
		for _ in range(self.repeat_action):
			if 'NoFrameskip' in self.task:
				s,r,done,step_info = self.skip_frames(a)
			else:
				s,r,done,step_info = self.env.step(a) # don't skip the frames if it is skipped already by gym
			frames.append(s)
			rewards.append(r)
		total_rewards = sum(rewards)
		return frames, total_rewards, done, step_info 

	def render(self):
		self.env.render()

