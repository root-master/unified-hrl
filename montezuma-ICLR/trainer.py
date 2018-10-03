import random
import numpy as np
import copy 
import pickle 
from image_processing import *
from memory import Experience, ExperienceMeta
from Environment import Environment
import time

class IntrinsicMotivation():
	def __init__(self,
				 env=None,
				 controller=None,
				 experience_memory=None,
				 image_processor=None,
				 subgoal_discovery=None,
				 **kwargs):

		self.env = env
		self.controller = controller
		self.experience_memory = experience_memory
		self.image_processor = image_processor
		self.subgoal_discovery = subgoal_discovery

		self.testing_env = Environment(task=self.env.task) # testing environment
		self.testing_scores = [] # record testing scores
		self.epsilon_testing = 0.05
		self.max_steps_testing = 200
		# parameters
		self.max_iter = 2500000
		self.controller_target_update_freq = 10000
		self.learning_freq = 4
		self.learning_starts = 50000
		self.save_model_freq = 50000
		self.test_freq = 10000
		self.subgoal_discovery_freq = 50000
		self.epsilon_start = 1.0
		self.epsilon_end = 0.1
		self.epsilon = self.epsilon_start
		self.epsilon_annealing_steps = 1000000
		self.repeat_noop_action = 0
		self.save_results_freq = 100000

		self.__dict__.update(kwargs) # updating input kwargs params 

		# counters
		self.step = 0 
		self.game_episode = 0
		self.intrinsic_motivation_learning_episode = 0 

		# learning variables
		self.episode_rewards = 0.0 # including step cost 
		self.episode_scores = 0.0
		# keeping records of the performance
		self.episode_rewards_list = []
		self.episode_scores_list = []

		print('init: Intrinsic Motivation Trainer --> OK')

	def train(self):
		print('-'*60)
		print('-'*60)
		print('PHASE I: Initial Intrinsic Motivation Learning and Subgoal Discovery')
		print('Purpose 1) Training Controller to reach random locations')
		print('Purpose 2) Discovering subgoals')
		print('-'*60)
		print('-'*60)

		# reset
		print('-'*60)
		print('game episode: ', self.game_episode)
		print('time step: ', self.step)
		S = self.env.reset()
		s = four_frames_to_4_84_84(S)
		man_mask = self.image_processor.get_man_mask(S)
		man_loc = get_man_xy_np_coordinate(man_mask)
		g_id, subgoal_mask = self.image_processor.sample_from_random_subgoal_set() # random g
		print('new subgoal assigned, g_id = ', g_id)
		subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask)
		g = single_channel_frame_to_1_84_84(subgoal_frame)
		for t in range(self.max_iter+1):
			self.step = t
			if t < self.learning_starts:
				a = self.env.action_space.sample()
			else:
				a = self.epsilon_greedy(s,g)

			old_lives = self.env.lives()
			SP, r, terminal, step_info = self.env.step(a)
			new_lives = self.env.lives()
			self.episode_scores += r	
			sp = four_frames_to_4_84_84(SP)
			man_mask = self.image_processor.get_man_mask(SP)
			man_loc = get_man_xy_np_coordinate(man_mask)
			intrinsic_done_task = is_man_inside_subgoal_mask(man_mask, subgoal_mask)
			
			# outlier for the subgoal discovery
			if r > 0:
				print('############# found an outlier ###############')
				self.subgoal_discovery.push_outlier(man_loc)
			else:
				r = -0.1 # small negative reward

			if intrinsic_done_task:
				intrinsic_done = 1 # binary mask		
				print('succesful intrinsic motivation learning to g_id = ', g_id)
				r_tilde = +1.0
				self.intrinsic_motivation_learning_episode += 1				
			else:
				intrinsic_done = 0
				r_tilde = -0.1 # small negetive reward to motivate agent to solve task
			
			if new_lives < old_lives:
				print('agent died, current lives = ', new_lives)
				r = -1.0
				r_tilde = -1.0 # dying reward

			if r > 100: # it means solving room #1 which in our paper is equivalent to task comelition
				task_done = True
				done = 1 # binary mask for done
				print('The room #1 task is completed, needs to reset!')
			else:
				task_done = False
				done = 0

			if terminal:
				print('agent terminated, end of episode') 
				r = -10.0

			self.episode_rewards += r # including negative rewards for death

			r = np.clip(r, -1.0, 1.0)
			experience = Experience(s, g, g_id, a, r, r_tilde, sp, intrinsic_done, done, man_loc)
			self.experience_memory.push(experience)

			s = copy.deepcopy(sp)
			self.anneal_epsilon()

			if intrinsic_done_task: # reset subgoal when intrinsic motivation task is accomplished
				g_id, subgoal_mask = self.image_processor.sample_from_random_subgoal_set() # random g
				print('new subgoal assigned, g_id = ', g_id)
				subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask)
				g = single_channel_frame_to_1_84_84(subgoal_frame)

			if (new_lives < old_lives) and not terminal and self.repeat_noop_action>0:
				for _ in range(self.repeat_noop_action): # do 20 nothing actions to ignore post-death
					S,_,_,_ = self.env.step(0)
				s = four_frames_to_4_84_84(S)

			if terminal or task_done:
				self.episode_scores_list.append(self.episode_scores)
				self.episode_rewards_list.append(self.episode_rewards)
				self.game_episode += 1
				self.episode_rewards = 0.0
				self.episode_scores = 0.0				
				print('-'*60)
				print('game episode: ', self.game_episode)
				print('time step: ', self.step)
				S = self.env.reset() # reset S
				s = four_frames_to_4_84_84(S) # get s
				man_mask = self.image_processor.get_man_mask(S) # man's mask
				man_loc = get_man_xy_np_coordinate(man_mask) # man's np location
				g_id, subgoal_mask = self.image_processor.sample_from_random_subgoal_set() # id and mask of random subgoal
				print('new subgoal assigned, g_id = ', g_id)
				subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask) #subgoal frame
				g = single_channel_frame_to_1_84_84(subgoal_frame)

			if (t>self.learning_starts) and (t % self.learning_freq == 0):
				self.controller.update_w()				

			if (t>0) and (t % self.subgoal_discovery_freq==0): # find centroids
				X = self.experience_memory.get_man_positions()
				self.subgoal_discovery.feed_data(X)
				self.subgoal_discovery.find_kmeans_clusters()
				results_file_path = './results/subgoal_discovery_step_' + str(t) + '.pkl'
				self.subgoal_discovery.save_results(results_file_path=results_file_path)

			if (t>self.learning_starts) and (t % self.test_freq == 0): # test controller's performance
				self.test()

			if (t>0) and (t % self.save_model_freq == 0): # save controller model
				model_save_path = './models/controller_step_' + str(t) + '.model'
				self.controller.save_model(model_save_path)
				print('saving model, steps = ', t)

			if (t>0) and (t % self.save_results_freq == 0):
				results_file_path = './results/performance_results_' + str(t) + '.pkl'
				with open(results_file_path, 'wb') as f: 
					pickle.dump([self.episode_scores_list,self.episode_rewards_list,self.testing_scores], f)

			if (t>self.learning_starts) and (t % self.controller_target_update_freq == 0):
				self.controller.update_target_params()

			
	def test(self):
		self.total_score_testing = 0
		self.testing_task_done = False
		# subgoals_order_before_key = [0,1,4,5,3,2]
		# key = [6]
		# if random.random() < 0.5: # flip a coin		
		# 	subgoals_order_after_key = [2,3,5,4,1,0,8]
		# else:
		# 	subgoals_order_after_key = [2,3,5,4,1,0,10]
		# subgoal_orders = subgoals_order_before_key + key + subgoals_order_after_key
		subgoals_order_before_key = [0,1,2,6,7,8,5,4,3,9]
		key = [9]
		subgoals_order_after_key = [3,4,5,8,7,6,2,1,0]
		if random.random() < 0.5: # flip a coin		
			door = [10]
		else:
			door = [11]
		subgoal_orders = subgoals_order_before_key + key + subgoals_order_after_key + door
		print('testing the controller')
		self.S_test = self.testing_env.reset()
		for g_id in subgoal_orders:
			self.test_reaching_subgoal(g_id=g_id)
			if self.testing_task_done:
				print('testing is succesful!')
				break
		if not self.testing_task_done:
			print('testing is not succesful!')

		self.testing_scores.append(self.total_score_testing) 
		print('test score: ', self.total_score_testing)

	def test_reaching_subgoal(self,g_id=0):
		S = self.S_test
		s = four_frames_to_4_84_84(S)
		man_mask = self.image_processor.get_man_mask(S)
		man_loc = get_man_xy_np_coordinate(man_mask)
		subgoal_mask = self.image_processor.random_subgoals_set[g_id]
		print('testing to reach subgoal, g_id = ', g_id)
		subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask)
		g = single_channel_frame_to_1_84_84(subgoal_frame)
		for t in range(self.max_steps_testing):
			a = self.epsilon_greedy_testing(s,g)
			old_lives = self.testing_env.lives()
			SP, r, terminal, step_info = self.testing_env.step(a)
			self.total_score_testing += r
			new_lives = self.testing_env.lives()
			self.episode_rewards += r	
			sp = four_frames_to_4_84_84(SP)
			man_mask = self.image_processor.get_man_mask(SP)
			man_loc = get_man_xy_np_coordinate(man_mask)
			intrinsic_done_task = is_man_inside_subgoal_mask(man_mask, subgoal_mask)
			
			# outlier for the subgoal discovery
			if r > 0:
				print('############# found an outlier - test time ###############')
				self.subgoal_discovery.push_outlier(man_loc)
				self.episode_scores += r
			
			if new_lives < old_lives:
				print('agent died, current lives = ', new_lives)

			if r > 100: # it means solving room #1 which in our paper is equivalent to task comelition
				self.testing_task_done = True
				break
			else:
				self.testing_task_done = False

			if terminal:
				print('agent terminated, end of episode') 
				self.S_test = self.testing_env.reset()
				break

			if intrinsic_done_task:
				print('succesful intrinsic motivation learning to g_id = ', g_id)
				self.S_test = SP
				break
			
			s = copy.deepcopy(sp)
			self.S_test = SP

			if (new_lives < old_lives) and not terminal and self.repeat_noop_action > 0:
				for _ in range(self.repeat_noop_action): # do 40 nothing actions to ignore post-death
					S,_,_,_ = self.env.step(0)
				self.S_test = S
				s = four_frames_to_4_84_84(S)


	def anneal_epsilon(self):
		if self.step < self.epsilon_annealing_steps:
			slop = (self.epsilon_start-self.epsilon_end)/self.epsilon_annealing_steps
			self.epsilon = self.epsilon_start - slop*self.step

	def epsilon_greedy(self,s,g):
		if random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return self.controller.get_best_action(s,g)

	def epsilon_greedy_testing(self,s,g):
		if random.random() < self.epsilon_testing:
			return self.env.action_space.sample()
		else:
			return self.controller.get_best_action(s,g)

class MetaControllerController():
	def __init__(self,
				env=None,
				controller=None,
				meta_controller = None,
				experience_memory=None,
				meta_controller_experience_memory=None,
				image_processor=None,
				subgoal_discovery=None,
				**kwargs):
		self.env = env
		self.controller = controller
		self.meta_controller = meta_controller
		self.experience_memory = experience_memory
		self.meta_controller_experience_memory = meta_controller_experience_memory
		self.image_processor = image_processor
		self.subgoal_discovery = subgoal_discovery
		self.G = image_processor.discovered_subgoals_set

		self.testing_env = Environment(task=self.env.task) # testing environment
		self.epsilon_testing = 0.05
		self.epsilon_testing_meta = 0.1
		self.max_steps_testing = 200
		# parameters
		self.max_iter = 2500000
		self.controller_target_update_freq = 10000
		self.meta_controller_target_update_freq = 10000
		self.learning_freq = 4
		self.meta_learning_freq = 20
		self.learning_starts = 50000
		self.save_model_freq = 50000
		self.test_freq = 10000
		self.meta_controller_test_freq = 50000
		self.subgoal_discovery_freq = 50000
		self.epsilon_start = 1.0
		self.epsilon_end = 0.1
		self.epsilon = self.epsilon_start
		self.epsilon_annealing_steps = 1000000
		self.repeat_noop_action = 0
		self.save_results_freq = 100000
		# counters
		self.step = 0 
		self.game_episode = 0
		self.meta_episode = 0
		self.intrinsic_motivation_learning_episode = 0 
		self.max_episode_steps = 1000
		self.episode_steps = 0
		# learning variables
		self.episode_rewards = 0.0 # meta controller episode return including step cost 
		self.game_episode_rewards = 0.0 # game episode return including step cost
		self.episode_scores = 0.0 # meta controller episode score
		self.game_episode_scores = 0.0 # game episode score
		# keeping records of the performance
		self.episode_rewards_list = []
		self.episode_scores_list = []
		self.game_episode_scores_list = []
		self.game_episode_rewards_list = []

		self.train_assignment_subgoal_count = [0]*len(self.G)
		self.testing_assignment_subgoal_count = [0]*len(self.G)

		self.train_success_subgoal_count = [0]*len(self.G)
		self.testing_success_subgoal_count = [0]*len(self.G)
		self.meta_controller_testing_scores = []
		self.testing_scores = [] # record testing scores for controller
		self.__dict__.update(kwargs) # updating input kwargs params
		print('init: Meta Controller - Controller Trainer --> OK')

	def train(self):
		print('-'*60)
		print('-'*60)
		print('PHASE II:  Intrinsic Motivation Learning and Meta Learning')
		print('Purpose 1) Training Controller to reach discovered locations')
		print('Purpose 2) Training Meta Controller to choose the best subgoal')
		print('-'*60)
		print('-'*60)
		# reset
		print('-'*60)
		print('game episode: ', self.game_episode)
		print('time step: ', self.step)
		S = self.env.reset()
		s = four_frames_to_4_84_84(S)
		s0 = copy.deepcopy(s)
		man_mask = self.image_processor.get_man_mask(S)
		man_loc = get_man_xy_np_coordinate(man_mask)
		g_id, subgoal_mask = self.get_subgoal(s)
		g0 = copy.deepcopy(g_id)
		self.train_assignment_subgoal_count[g0]+=1
		print('new subgoal assigned = ', self.image_processor.discovered_subgoal_meaning_set[g_id])
		subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask)
		g = single_channel_frame_to_1_84_84(subgoal_frame)

		for t in range(self.max_iter+1):
			self.step = t
			self.episode_steps += 1
			if t < self.learning_starts:
				a = self.env.action_space.sample()
			else:
				a = self.epsilon_greedy(s,g)

			old_lives = self.env.lives()
			SP, r, terminal, step_info = self.env.step(a)
			new_lives = self.env.lives()
			self.episode_scores += r
			self.game_episode_scores += r	
			sp = four_frames_to_4_84_84(SP)
			man_mask = self.image_processor.get_man_mask(SP)
			man_loc = get_man_xy_np_coordinate(man_mask)
			intrinsic_done_task = is_man_inside_subgoal_mask(man_mask, subgoal_mask)
			
			# outlier for the subgoal discovery
			if r > 0:
				print('############# found an outlier ###############')
				self.subgoal_discovery.push_outlier(man_loc)
			else:
				r = -0.1 # small negative reward

			if intrinsic_done_task:
				intrinsic_done = 1 # binary mask		
				print('succesful intrinsic motivation learning to ', self.image_processor.discovered_subgoal_meaning_set[g_id])
				r_tilde = +1.0
				self.intrinsic_motivation_learning_episode += 1
				self.train_success_subgoal_count[g0] += 1			
			else:
				intrinsic_done = 0
				r_tilde = -0.1 # small negetive reward to motivate agent to solve task
			
			if new_lives < old_lives:
				print('agent died, current lives = ', new_lives)
				r = -1.0
				r_tilde = -1.0 # dying reward

			if r > 100: # it means solving room #1 which in our paper is equivalent to task comelition
				task_done = True
				done = 1 # binary mask for done
				print('The room #1 task is completed, needs to reset!')
			else:
				task_done = False
				done = 0

			if terminal:
				print('agent terminated, end of episode') 
				r = -10.0

			self.episode_rewards += r # including negative rewards for death
			self.game_episode_rewards += r

			r = np.clip(r, -1.0, 1.0)
			experience = Experience(s, g, g_id, a, r, r_tilde, sp, intrinsic_done, done, man_loc)
			self.experience_memory.push(experience)

			s = copy.deepcopy(sp)
			self.anneal_epsilon()

			if intrinsic_done_task or (self.episode_steps>self.max_episode_steps): # reset subgoal when intrinsic motivation task is accomplished
				self.meta_episode += 1
				s1 = copy.deepcopy(s)
				meta_controller_experience = ExperienceMeta(s0,g0,self.episode_rewards,s1,done)
				s0 = copy.deepcopy(s1)
				self.meta_controller_experience_memory.push(meta_controller_experience)
				self.episode_rewards_list.append(self.episode_rewards)
				self.episode_scores_list.append(self.episode_scores)
				self.episode_rewards = 0.0
				self.episode_scores = 0.0
				self.episode_steps = 0
				self.meta_episode = 0
				g_id, subgoal_mask = self.get_subgoal(s) # get g
				print('new subgoal assigned = ', self.image_processor.discovered_subgoal_meaning_set[g_id])
				subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask)
				g = single_channel_frame_to_1_84_84(subgoal_frame)
				g0 = copy.deepcopy(g_id)
				self.train_assignment_subgoal_count[g0] += 1

			if (new_lives < old_lives) and not terminal and self.repeat_noop_action>0:
				for _ in range(self.repeat_noop_action): # do 20 nothing actions to ignore post-death
					S,_,_,_ = self.env.step(0)
				s = four_frames_to_4_84_84(S)
				s0 = copy.deepcopy(s)

			if terminal or task_done:
				self.game_episode_scores_list.append(self.game_episode_scores)
				self.game_episode_rewards_list.append(self.game_episode_rewards)
				self.game_episode += 1
				self.episode_rewards = 0.0
				self.game_episode_rewards = 0.0
				self.episode_scores = 0.0
				self.game_episode_scores = 0.0
				self.episode_steps = 0				
				print('-'*60)
				print('game episode: ', self.game_episode)
				print('time step: ', self.step)
				S = self.env.reset() # reset S
				s = four_frames_to_4_84_84(S) # get s
				s0 = copy.deepcopy(s)
				man_mask = self.image_processor.get_man_mask(S) # man's mask
				man_loc = get_man_xy_np_coordinate(man_mask) # man's np location
				g_id, subgoal_mask = self.get_subgoal(s)
				print('new subgoal assigned = ', self.image_processor.discovered_subgoal_meaning_set[g_id])
				subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask) #subgoal frame
				g = single_channel_frame_to_1_84_84(subgoal_frame)
				g0 = copy.deepcopy(g_id)
				self.train_assignment_subgoal_count[g0] += 1

			if (t>self.learning_starts) and (t % self.learning_freq == 0):
				self.controller.update_w()				

			if (t>self.learning_starts) and (t % self.meta_learning_freq == 0):
				self.meta_controller.update_w()				

			if (t>0) and (t % self.subgoal_discovery_freq==0): # find centroids
				X = self.experience_memory.get_man_positions()
				self.subgoal_discovery.feed_data(X)
				self.subgoal_discovery.find_kmeans_clusters()
				results_file_path = './results_phase_2/subgoal_discovery_step_' + str(t) + '.pkl'
				self.subgoal_discovery.save_results(results_file_path=results_file_path)

			if (t>self.learning_starts) and (t % self.test_freq == 0): # test controller's performance
				self.test_controller()

			if (t>self.learning_starts) and (t % self.meta_controller_test_freq == 0):
				self.test_meta_controller()

			if (t>0) and (t % self.save_model_freq == 0): # save controller model
				model_save_path = './models_phase_2/controller_step_' + str(t) + '.model'
				self.controller.save_model(model_save_path)
				model_save_path = './models_phase_2/meta_controller_step_' + str(t) + '.model'
				self.meta_controller.save_model(model_save_path)
				print('saving models, steps = ', t)

			if (t>0) and (t % self.save_results_freq == 0):
				results_file_path = './results_phase_2/performance_results_' + str(t) + '.pkl'
				with open(results_file_path, 'wb') as f: 
					pickle.dump([self.episode_scores_list,
								 self.episode_rewards_list,
								 self.game_episode_scores_list,
								 self.game_episode_rewards_list,
								 self.train_assignment_subgoal_count,
								 self.train_success_subgoal_count,
								 self.testing_scores,
								 self.meta_controller_testing_scores], f)
			
			if (t>self.learning_starts) and (t % self.controller_target_update_freq == 0):
				self.controller.update_target_params()
			if (t>self.learning_starts) and (t % self.meta_controller_target_update_freq == 0):
				self.meta_controller.update_target_params()

	def get_subgoal(self,s):
		if self.step < self.learning_starts:
			g_id, subgoal_mask = self.image_processor.sample_from_discovered_subgoal_set()
		else:
			g_id = self.epsilon_greedy_meta_controller(s)
			subgoal_mask = self.G[g_id]
		return g_id, subgoal_mask

	def test_meta_controller(self):
		self.max_total_score_testing = 0
		print('testing the meta controller')
		meta_test_episode = 0
		self.S_test = self.testing_env.reset()
		self.task_done = False
		self.intrinsic_done_task = False
		while(meta_test_episode<2): # let agent plays 10 episodes 		
			self.total_score_testing = 0
			if self.task_done:
				print('meta controller testing is succesful!')
				break
			self.terminal = False
			while not self.terminal:
				S = self.S_test
				s = four_frames_to_4_84_84(S)
				g_id = self.epsilon_greedy_meta_controller_testing(s)	
				self.intrinsic_done_task = False
				self.test_reaching_subgoal(g_id=g_id)
				if self.terminal or self.task_done:
					meta_test_episode += 1
					break

			self.max_total_score_testing = max([self.max_total_score_testing,self.total_score_testing])

		if not self.task_done:
			print('meta controller testing is not succesful!')
		self.meta_controller_testing_scores.append(self.max_total_score_testing) 
		print('meta controller best test score: ', self.max_total_score_testing)

	def test_controller(self):
		self.total_score_testing = 0
		self.testing_task_done = False
		if len(self.G)==6:
			subgoals_order_before_key = [0,2,1]
			key = [3]
			subgoals_order_after_key = [1,2,0]
			if random.random() < 0.5: # flip a coin		
				door = [4]
			else:
				door = [5]

		if len(self.G)==12:
			subgoals_order_before_key = [0,1,2,6,7,8,5,4,3,9]
			key = [9]
			subgoals_order_after_key = [3,4,5,8,7,6,2,1,0]
			if random.random() < 0.5: # flip a coin		
				door = [10]
			else:
				door = [11]
		
		
		subgoal_orders = subgoals_order_before_key + key + subgoals_order_after_key + door
		print('testing the controller')
		self.S_test = self.testing_env.reset()
		for g_id in subgoal_orders:
			self.test_reaching_subgoal(g_id=g_id)
			if self.testing_task_done:
				print('testing is succesful!')
				break
		if not self.testing_task_done:
			print('testing is not succesful!')

		self.testing_scores.append(self.total_score_testing) 
		print('test score: ', self.total_score_testing)

	def test_reaching_subgoal(self,g_id=0):
		S = self.S_test
		s = four_frames_to_4_84_84(S)
		man_mask = self.image_processor.get_man_mask(S)
		man_loc = get_man_xy_np_coordinate(man_mask)
		subgoal_mask = self.G[g_id]
		print('testing to reach subgoal, g_id = ', g_id)
		subgoal_frame = self.image_processor.create_mask_frame(subgoal_mask)
		g = single_channel_frame_to_1_84_84(subgoal_frame)
		self.intrinsic_done_task = False
		self.terminal = False
		self.task_done = False
		for t in range(self.max_steps_testing):
			a = self.epsilon_greedy_testing(s,g)
			old_lives = self.testing_env.lives()
			SP, r, terminal, step_info = self.testing_env.step(a)
			self.total_score_testing += r
			new_lives = self.testing_env.lives()
			sp = four_frames_to_4_84_84(SP)
			man_mask = self.image_processor.get_man_mask(SP)
			man_loc = get_man_xy_np_coordinate(man_mask)
			intrinsic_done_task = is_man_inside_subgoal_mask(man_mask, subgoal_mask)
			self.intrinsic_done_task = intrinsic_done_task
			self.terminal = terminal
			# outlier for the subgoal discovery
			if r > 0:
				print('############# found an outlier - test time ###############')
				self.subgoal_discovery.push_outlier(man_loc)
			
			if new_lives < old_lives:
				print('agent died, current lives = ', new_lives)

			if r > 100: # it means solving room #1 which in our paper is equivalent to task comelition
				self.testing_task_done = True
				self.task_done = True
				break
			else:
				self.testing_task_done = False

			if terminal:
				print('agent terminated, end of episode') 
				self.S_test = self.testing_env.reset()
				break

			if intrinsic_done_task:
				print('succesful intrinsic motivation learning to g_id = ', g_id)
				self.S_test = SP
				break
			
			s = copy.deepcopy(sp)
			self.S_test = SP

			if (new_lives < old_lives) and not terminal and self.repeat_noop_action > 0:
				for _ in range(self.repeat_noop_action): # do 40 nothing actions to ignore post-death
					S,_,_,_ = self.env.step(0)
				self.S_test = S
				s = four_frames_to_4_84_84(S)

	def anneal_epsilon(self):
		if self.step < self.epsilon_annealing_steps:
			slop = (self.epsilon_start-self.epsilon_end)/self.epsilon_annealing_steps
			self.epsilon = self.epsilon_start - slop*self.step

	def epsilon_greedy(self,s,g):
		if random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return self.controller.get_best_action(s,g)

	def epsilon_greedy_meta_controller(self,s):
		if random.random() < self.epsilon:
			return random.randint(0, len(self.G)-1)
		else:
			return self.meta_controller.get_best_option(s)

	def epsilon_greedy_meta_controller_testing(self,s):
		if random.random() < self.epsilon_testing_meta:
			return random.randint(0, len(self.G)-1)
		else:
			return self.meta_controller.get_best_option(s)

	def epsilon_greedy_testing(self,s,g):
		if random.random() < self.epsilon_testing:
			return self.env.action_space.sample()
		else:
			return self.controller.get_best_action(s,g)




