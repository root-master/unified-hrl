import copy
from random import random, randint

class RandomWalk():
	def __init__(self,
				env=None,
				subgoal_discovery=None,
				experience_memory=None,
				**kwargs):
		self.env = env
		self.max_steps = 200
		self.max_episodes = 100
		self.subgoal_discovery = subgoal_discovery
		self.experience_memory = experience_memory
		self.subgoal_dicovery_freq = 25
		self.__dict__.update(kwargs)

	def walk(self):
		for i in range(self.max_episodes):
			s = self.env.reset()
			for j in range(self.max_steps):			
				a = self.env.action_space.sample()
				sp, r, terminal, step_info = self.env.step(a)
				e = (s,a,r,sp)
				self.experience_memory.push(e)
				if r>0:
					self.subgoal_discovery.push_outlier(sp)
				if terminal:
					break
				s = copy.copy(sp)

			if i>0 and i%25 == 0:
				self.subgoal_discovery.find_kmeans_clusters()
				self.subgoal_discovery.report()

	def walk_and_find_doorways(self):
		for i in range(self.max_episodes):
			s = self.env.reset()
			for j in range(self.max_steps):			
				a = self.env.action_space.sample()
				sp, r, terminal, step_info = self.env.step(a)
				e = (s,a,r,sp)
				self.subgoal_discovery.push_doorways(e)
				if terminal:
					break
				s = copy.copy(sp)

class PretrainController():
	def __init__(self,
				env=None,
				controller=None,
				subgoals=None):
		self.env = env
		self.controller = controller
		self.subgoals = subgoals
		self.max_episodes = 100000
		self.max_steps = 100
		self.epsilon = 0.2
		self.nA = 4
		self.ng = 6
		self.gamma = 0.99
		self.lr = 0.001
		self.pretrain_all_subgoals_success_list = []

	def train(self):
		for g_id in range(self.ng):
			self.pretrain_success_subgoal = []
			self.train_for_one_goal(g_id)
			self.pretrain_all_subgoals_success_list.append(self.pretrain_success_subgoal)

	def train_for_one_goal(self,g_id):
		g = self.subgoals[g_id]
		for i in range(self.max_episodes):
			s = self.env.reset()
			for j in range(self.max_steps):	
				Q = self.controller.Q.compute_Q(s,g_id)
				a = self.epsilon_greedy(Q)
				sp,r,done,info = self.env.step(a)
				Qp = self.controller.Q.compute_Qp(sp, g_id)
				ap = self.epsilon_greedy(Qp)
				if self.env.state_before_passing_doorway in g:
					terminal = True
					done_mask = 1
					r_tilde = 1.0
					print('intrinsic motivation is solved in episode: ', i)
				else:
					terminal = False
					done_mask = 0
					r_tilde = min(-0.1,r)
				delta = r_tilde + (1 - done_mask) * self.gamma * Qp[0,ap] - Q[0,a]
				delta = delta * self.lr
				self.controller.Q.update_w(a,delta)
				# self.controller.Q.update_Q_to_Qp()
				s = copy.copy(sp)
				a = copy.copy(ap)		
				if done:
					print('solved the rooms task in episode :', i)
				
				if terminal or done:
					break
			self.pretrain_success_subgoal.append(done_mask)

	def epsilon_greedy(self, Q, test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.nA-1)
		else:
			return Q.argmax()


class MetaControllerController():
	def __init__(self,
				env=None,
				controller=None,
				meta_controller=None,
				subgoals=None):
		self.env = env
		self.controller = controller
		self.meta_controller = meta_controller
		self.subgoals = subgoals
		self.max_episodes = 100000
		self.max_steps = 200
		self.max_steps_controller = 25
		self.epsilon = 0.2
		self.ng = 6
		self.ns = 6
		self.gamma = 0.99
		self.lr = 0.001
		self.episode_rewards = []
		self.episode_success = []

	def train(self):
		for i in range(self.max_episodes):
			print('-'*60)
			self.R = 0
			self.s = self.env.reset()
			self.done = False
			self.t = 0
			while not self.done:
				S = self.meta_controller.get_meta_state(self.s,
										has_key=self.env.step_info['has_key'])
				Q = self.meta_controller.Q.compute_Q(S)
				g_id = self.epsilon_greedy(Q)
				self.play(g_id)
				if self.done:
					print('solved the rooms task in episode :', i)
					done_mask = 1
				else:
					done_mask = 0

				if self.terminal or self.done:
					print('reached to the subgoal', g_id)					
					SP = self.meta_controller.get_meta_state(self.s,
										has_key=self.env.step_info['has_key'])
					QP = self.meta_controller.Q.compute_QP(SP)
					g_id_prime = self.epsilon_greedy(QP)
					delta = self.R + self.gamma * (1 - done_mask) * QP[0,g_id_prime] - Q[0,g_id]
					delta = delta * self.lr
					self.meta_controller.Q.update_w(g_id,delta)

				if self.t > self.max_steps:
					break

			self.episode_rewards.append(self.R)
			self.episode_success.append(done_mask)

	def play(self,g_id):
		s = self.s
		g = self.subgoals[g_id]
		for j in range(self.max_steps_controller):	
			Q = self.controller.Q.compute_Q(s,g_id)
			a = self.epsilon_greedy(Q,test=True)
			sp,r,done,info = self.env.step(a)
			self.R += r
			self.done = done
			if self.env.state_before_passing_doorway in g:
				self.terminal = True
			else:
				self.terminal = False
			self.s = copy.copy(sp)
			self.t += 1
			if self.terminal or self.done:
				break

	def epsilon_greedy(self, Q,test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.ng-1)
		else:
			return Q.argmax()

