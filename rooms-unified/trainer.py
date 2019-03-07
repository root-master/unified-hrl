import copy
from random import random, randint
import pickle
from Model import VanillaRLModel
from tqdm import tqdm

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
		print('#'*60)
		print('Random walk for Subgoal Discovery')
		print('#'*60)
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
				# self.subgoal_discovery.report()

	def walk_and_find_doorways(self):
		print('#'*60)
		print('Random walk for Doorways Type Subgoal Discovery')
		print('#'*60)
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
				subgoal_discovery=None):
		self.env = env
		self.controller = controller
		self.subgoal_discovery = subgoal_discovery
		self.subgoals = subgoal_discovery.G
		self.max_episodes = 100000+1
		self.max_steps = 100
		self.epsilon = 0.2
		self.nA = 4
		self.ng = len(self.subgoals)
		self.gamma = 0.99
		self.lr = 0.001

	def train(self):
		print('#'*60)
		print('Pretraining Controller in Hierarchcial Reinforcement Leaning')
		print('#'*60)
		self.success_rate = {}
		for g_id in range(self.ng):
			self.success_rate[g_id] = 0.0
			self.success_intrinsic_motivation = 0
			print('subgoal # ', g_id)
			self.pretrain_success_subgoal = []
			self.train_for_one_goal(g_id)
			self.success_rate[g_id] = self.success_intrinsic_motivation/self.max_episodes
			print('average success for subgoal # ',g_id,' = ', self.success_rate[g_id])
			results_file_path = './results/pretraining_contoller_performance_results_K_' + str(self.ng) + '_subgoal_' + str(g_id) + '.pkl'
			with open(results_file_path, 'wb') as f: 
				pickle.dump(self.pretrain_success_subgoal, f)

	def train_for_one_goal(self,g_id):
		g = self.subgoals[g_id]
		for i in tqdm(range(self.max_episodes)):
			s = self.env.reset()
			for j in range(self.max_steps):	
				Q = self.controller.Q.compute_Q(s,g_id)
				a = self.epsilon_greedy_controller(Q)
				sp,r,done,info = self.env.step(a)
				Qp = self.controller.Q.compute_Qp(sp, g_id)
				ap = self.epsilon_greedy_controller(Qp)
				
				terminal = False
				if g in self.subgoal_discovery.outliers:
					if sp==g:
						terminal = True
				elif g in self.subgoal_discovery.centroid_subgoals:
					if self.subgoal_discovery.predict_closest_cluster_index(sp) == \
						self.subgoal_discovery.predict_closest_cluster_index(g):
						terminal = True

				if terminal:
					done_mask = 1
					r_tilde = 1.0
					# print('intrinsic motivation is solved in episode: ', i)
				else:
					terminal = False
					done_mask = 0
					r_tilde = min(-0.1,r)
				delta = r_tilde + (1 - done_mask) * self.gamma * Qp[0,ap] - Q[0,a]
				delta = delta * self.lr
				self.controller.Q.update_w(a,delta)
				s = copy.copy(sp)
				# if done:
				# 	print('solved the rooms task in episode :', i)
				
				if terminal or done:
					self.success_intrinsic_motivation += 1
					break
			self.pretrain_success_subgoal.append(done_mask)

	def epsilon_greedy_controller(self, Q, test=False):
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
				subgoal_discovery=None):
		self.env = env
		self.controller = controller
		self.meta_controller = meta_controller
		self.subgoal_discovery = subgoal_discovery
		self.subgoals = subgoal_discovery.G
		self.max_episodes = 100000+1
		self.max_steps = 200
		self.max_steps_controller = 25
		self.epsilon = 0.2
		self.epsilon_start = 0.2
		self.epsilon_end = 0.01
		self.epsilon_episode_end = 200
		self.ng = len(self.subgoals)
		self.ns = self.ng
		self.gamma = 0.99
		self.lr = 0.001
		self.episode_rewards = []
		self.episode_score = []
		self.episode_success = []
		self.save_results_freq = 1000
		self.nA = 4
		self.R = 0 # return
		self.G = 0 # score (r>0)
		self.t = 0
	def train(self):
		print('#'*60)
		print('Training Meta-controller in Hierarchcial Reinforcement Leaning')
		print('#'*60)
		for i in range(self.max_episodes):
			self.train_metacontroller()
			self.episode_rewards.append(self.R)
			self.episode_success.append(self.done_mask)
			self.episode_score.append(self.G)

			self.epsilon = self.epsilon_start + (self.epsilon_end-self.epsilon_start) * (i / self.epsilon_episode_end)
			self.epsilon = max(self.epsilon_end,self.epsilon)
			self.epsilon = min(self.epsilon_start,self.epsilon)

		results_file_path = './results/meta_contoller_performance_results_K_' + str(self.ng) + '.pkl'
		with open(results_file_path, 'wb') as f: 
			pickle.dump([self.episode_rewards,
						 self.episode_score,
						 self.episode_success], f)

	def train_metacontroller(self):
		self.R = 0
		self.G = 0
		self.s = self.env.reset()
		self.done = False
		self.t = 0
		while not self.done:
			S = self.meta_controller.get_meta_state(self.s,
									has_key=self.env.step_info['has_key'])
			Q = self.meta_controller.Q.compute_Q(S)
			g_id = self.epsilon_greedy_metacontroller(Q)
			self.play_controller(g_id)
			if self.done:
				# print('solved the rooms task in episode :', i)
				done_mask = 1
			else:
				done_mask = 0
			self.done_mask = done_mask

			if self.terminal or self.done:
				# print('reached to the subgoal', g_id)					
				SP = self.meta_controller.get_meta_state(self.s,
									has_key=self.env.step_info['has_key'])
				QP = self.meta_controller.Q.compute_QP(SP)
				g_id_prime = self.epsilon_greedy_metacontroller(QP)
				delta = self.R + self.gamma * (1 - done_mask) * QP[0,g_id_prime] - Q[0,g_id]
				delta = delta * self.lr
				self.meta_controller.Q.update_w(g_id,delta)

			if self.t > self.max_steps or self.done:
				break


	def play_controller(self,g_id):
		s = self.s
		g = self.subgoals[g_id]
		for j in range(self.max_steps_controller):	
			Q = self.controller.Q.compute_Q(s,g_id)
			a = self.epsilon_greedy_controller(Q,test=True)
			sp,r,done,info = self.env.step(a)
			self.R += r
			if r>0:
				self.G += r
			self.done = done

			terminal = False
			if g in self.subgoal_discovery.outliers:
				if sp==g:
					terminal = True
			elif g in self.subgoal_discovery.centroid_subgoals:
				if self.subgoal_discovery.predict_closest_cluster_index(sp) == \
					self.subgoal_discovery.predict_closest_cluster_index(g):
					terminal = True

			self.terminal = terminal

			s = copy.copy(sp)
			self.s = s
			self.t += 1
			if self.terminal or self.done:
				break

	def epsilon_greedy_metacontroller(self, Q,test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.ng-1)
		else:
			return Q.argmax()

	def epsilon_greedy_controller(self, Q, test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.nA-1)
		else:
			return Q.argmax()



class MetaControllerControllerUnified():
	def __init__(self,
				env=None,
				controller=None,
				meta_controller=None,
				subgoal_discovery=None,
				pretrain=True):
		self.env = env
		self.controller = controller
		self.meta_controller = meta_controller
		self.subgoal_discovery = subgoal_discovery
		self.subgoals = subgoal_discovery.G
		self.max_episodes = 100000+1
		self.max_steps = 200
		self.max_steps_controller = 50
		self.epsilon = 0.2
		self.epsilon_start = 0.2
		self.epsilon_end = 0.01
		self.epsilon_episode_end = 200
		self.ng = len(self.subgoal_discovery.G)
		self.ns = self.ng
		self.gamma = 0.99
		self.lr_meta = 0.001
		self.lr_cont = 0.0001
		self.episode_rewards = []
		self.episode_success = []
		self.episode_score = []
		self.save_results_freq = 1000
		self.nA = 4
		self.success_test = []
		self.return_test = []
		self.score_test = []
		self.G = 0 # score
		self.pretrain = pretrain
		if not self.pretrain:
			self.max_episodes = 1000000+1


	def train(self):
		print('#'*60)
		print('Training Meta-controller and Controller together')
		print('#'*60)
		for i in tqdm(range(self.max_episodes)):
			self.i = i
			self.train_metacontroller_controller()
			self.episode_rewards.append(self.R)
			self.episode_success.append(self.done_mask)
			self.episode_score.append(self.G)

			# self.epsilon = self.epsilon_start + (self.epsilon_end-self.epsilon_start) * (i / self.epsilon_episode_end)
			# self.epsilon = max(self.epsilon_end,self.epsilon)
			# self.epsilon = min(self.epsilon_start,self.epsilon)
			if i>0 and i % 100 == 0:
				self.test_metacontroller_controller()
				self.return_test.append(self.R)
				self.score_test.append(self.G)
				self.success_test.append(self.done_mask)

		results_file_path = './results/meta_contoller_controller_performance_results_K_' + str(self.ng) + '.pkl'
		if not self.pretrain:
			results_file_path = './results/meta_contoller_controller_no_pretrain_performance_results_K_' + str(self.ng) + '.pkl'

		with open(results_file_path, 'wb') as f: 
			pickle.dump([self.episode_rewards,
						self.episode_score,
						self.episode_success,
						self.return_test,
						self.score_test,
						self.success_test], f)

	def train_metacontroller_controller(self):
		self.R = 0
		self.G = 0
		self.s = self.env.reset()
		self.done = False
		self.t = 0
		self.terminal = False
		while not self.done:
			S = self.meta_controller.get_meta_state(self.s,
									has_key=self.env.step_info['has_key'])
			Q = self.meta_controller.Q.compute_Q(S)
			g_id = self.epsilon_greedy_metacontroller(Q)
			self.play_train_controller(g_id)
			if self.done:
				# print('solved the rooms task in episode :', i)
				done_mask = 1
			else:
				done_mask = 0
			self.done_mask = done_mask

			if self.terminal or self.done:
				# print('reached to the subgoal', g_id)					
				SP = self.meta_controller.get_meta_state(self.s,
									has_key=self.env.step_info['has_key'])
				QP = self.meta_controller.Q.compute_QP(SP)
				g_id_prime = self.epsilon_greedy_metacontroller(QP)
				delta = self.R + self.gamma * (1 - done_mask) * QP[0,g_id_prime] - Q[0,g_id]
				delta = delta * self.lr_meta
				self.meta_controller.Q.update_w(g_id,delta)

			if self.t > self.max_steps:
				break

	def test_metacontroller_controller(self):
		self.G = 0
		self.R = 0
		self.s = self.env.reset()
		self.done = False
		self.t = 0
		self.terminal = False
		while not self.done:
			S = self.meta_controller.get_meta_state(self.s,
									has_key=self.env.step_info['has_key'])
			Q = self.meta_controller.Q.compute_Q(S)
			g_id = self.epsilon_greedy_metacontroller(Q,test=True)
			self.play_greedy_controller(g_id)

			if self.done:
				done_mask = 1
			else:
				done_mask = 0
			self.done_mask = done_mask

			if self.t > self.max_steps or self.done:
				break


	def play_greedy_controller(self,g_id):
		s = self.s
		g = self.subgoals[g_id]
		for j in range(self.max_steps_controller):	
			Q = self.controller.Q.compute_Q(s,g_id)
			a = self.epsilon_greedy_controller(Q,test=True)
			sp,r,done,info = self.env.step(a)
			self.R += r
			self.done = done
			if r > 0:
				self.G += r

			terminal = False
			if g in self.subgoal_discovery.outliers:
				if sp==g:
					terminal = True
			elif g in self.subgoal_discovery.centroid_subgoals:
				if self.subgoal_discovery.predict_closest_cluster_index(sp) == \
					self.subgoal_discovery.predict_closest_cluster_index(g):
					terminal = True

			self.terminal = terminal
			s = copy.copy(sp)
			self.s = s
			self.t += 1
			if self.terminal or self.done:
				break		

	def play_train_controller(self,g_id):
		s = self.s
		g = self.subgoals[g_id]
		for j in range(self.max_steps):	
			Q = self.controller.Q.compute_Q(s,g_id)
			a = self.epsilon_greedy_controller(Q)
			sp,r,done,info = self.env.step(a)
			self.R += r
			if r>0:
				self.G += r
			self.done = done

			Qp = self.controller.Q.compute_Qp(sp, g_id)
			ap = self.epsilon_greedy_controller(Qp)
			
			terminal = False
			if g in self.subgoal_discovery.outliers:
				if sp==g:
					terminal = True
			elif g in self.subgoal_discovery.centroid_subgoals:
				if self.subgoal_discovery.predict_closest_cluster_index(sp) == \
					self.subgoal_discovery.predict_closest_cluster_index(g):
					terminal = True

			if terminal:
				done_mask = 1
				r_tilde = 1.0
				# print('intrinsic motivation is solved in episode ', self.i )
			else:
				done_mask = 0
				r_tilde = min(-0.1,r)

			self.terminal = terminal
			delta = r_tilde + (1 - done_mask) * self.gamma * Qp[0,ap] - Q[0,a]
			delta = delta * self.lr_cont
			self.controller.Q.update_w(a,delta)
			s = copy.copy(sp)
			self.s = s
			self.t += 1
			if self.terminal or self.done:
				break

	def epsilon_greedy_metacontroller(self, Q,test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.ng-1)
		else:
			return Q.argmax()

	def epsilon_greedy_controller(self, Q, test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.nA-1)
		else:
			return Q.argmax()


class VanillaRL():
	def __init__(self,
				env=None):
		self.env = env
		self.Q = VanillaRLModel()
		self.env = env
		self.max_episodes = 100000
		self.max_steps = 200
		self.epsilon = 0.2
		self.nA = 4
		self.gamma = 0.99
		self.lr = 0.001
		self.episode_rewards = []
		self.episode_success = []
		self.R = 0
		self.save_results_freq = 1000

	def train(self):
		print('#'*60)
		print('Training Vanilla Reinforcement Leaning')
		print('#'*60)
		for i in range(self.max_episodes):
			s = self.env.reset()
			self.R = 0
			for j in range(self.max_steps):	
				Q = self.Q.compute_Q(s)
				a = self.epsilon_greedy(Q)
				sp,r,done,info = self.env.step(a)
				self.R += r
				Qp = self.Q.compute_Qp(sp)
				ap = self.epsilon_greedy(Qp)
				done_mask = 1 if done else 0
				delta = r + (1 - done_mask) * self.gamma * Qp[0,ap] - Q[0,a]
				delta = delta * self.lr
				self.Q.update_w(a,delta)
				s = copy.copy(sp)
				if done:
					# print('solved the rooms task in episode :', i)
					break
			self.episode_rewards.append(self.R)
			self.episode_success.append(done_mask)

		results_file_path = './results/vanilla-RL_performance_results.pkl'
		with open(results_file_path, 'wb') as f: 
			pickle.dump([self.episode_rewards,self.episode_success], f)
					
	def epsilon_greedy(self, Q,test=False):
		if test:
			return Q.argmax()

		if random() < self.epsilon:
			return randint(0, self.nA-1)
		else:
			return Q.argmax()




