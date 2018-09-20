import random
import numpy as np
import copy 
import pickle 
from image_processing import *
from memory import Experience
from Environment import Environment
import time

class IntrinsicMotivation():
	def __init__(self,
				 env=None,
				 controller=None,
				 experience_memory=None,
				 image_processor=None,
				 subgoal_discory=None,
				 **kwargs):

		self.env = env
		self.controller = controller
		self.experience_memory = experience_memory
		self.image_processor = image_processor
		self.subgoal_discory = subgoal_discory

		self.testing_env = Environment(task=self.env.task) # testing environment
		self.testing_scores = [] # record testing scores
		self.epsilon_testing = 0.05
		self.max_steps_testing = 10000
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
		print('game epsiode: ', self.game_episode)
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
			self.epsiode_scores += r	
			sp = four_frames_to_4_84_84(SP)
			man_mask = self.image_processor.get_man_mask(SP)
			man_loc = get_man_xy_np_coordinate(man_mask)
			intrinsic_done_task = is_man_inside_subgoal_mask(man_mask, subgoal_mask)
			
			# outlier for the subgoal discovery
			if r > 0:
				print('############# found an outlier ###############')
				self.subgoal_discory.push_outlier(man_loc)
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
				self.episode_scores_list.append(self.epsiode_scores)
				self.episode_rewards_list.append(self.episode_rewards)
				self.game_episode += 1
				self.episode_rewards = 0.0
				self.epsiode_scores = 0.0				
				print('-'*60)
				print('game epsiode: ', self.game_episode)
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
				self.subgoal_discory.feed_data(X)
				self.subgoal_discory.find_kmeans_clusters()

			if (t>self.learning_starts) and (t % self.test_freq == 0): # test controller's performance
				self.test()

			if (t>0) and (t % self.save_model_freq == 0): # save controller model
				model_save_path = './models/controller_step_' + str(t) + '.model'
				self.controller.save_model(model_save_path)
				print('saving model, steps = ', t)
			
	def test(self):
		self.total_score_testing = 0
		self.testing_task_done = False
		subgoals_order_before_key = [0,1,4,5,3,2]
		key = [6]
		if random.random() < 0.5: # flip a coin		
			subgoals_order_after_key = [1,0,10]
		else:
			subgoals_order_after_key = [1,0,8]
		subgoal_orders = subgoals_order_before_key + key + subgoals_order_after_key
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
				self.subgoal_discory.push_outlier(man_loc)
				self.epsiode_scores += r
			
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
				meta_controller = None,
				controller=None,
				expereince_memory=None,
				image_processor=None,
				subgoal_discory=None,
				**kwargs):

		self.__dict__.update(kwargs) # updating input kwargs params
		print('init: Meta Controller - Controller Trainer --> OK')

	def train(self):
		print('-'*60)
		print('-'*60)
		print('PHASE II: Secondary Intrinsic Motivation Learning and Meta Learning')
		print('Purpose 1) Training Controller to reach discovered locations')
		print('Purpose 2) Training Meta Controller to choose the best subgoal')
		print('-'*60)
		print('-'*60)


