import time
import gym
import numpy as np
from copy import copy
import random

from Model import Model_1

from ExperienceReplayMemory import ExperienceReplayMemory

from Model import Model_1, QTable
# model = Model_1()
q_table_is_used = True

if q_table_is_used:
	model = QTable()
else:
	model = Model_1()

RENDER = False
VERBOSE = True

from gym_rooms.envs import *
environment = 'Rooms-v0'
env = gym.make(environment)

max_steps = 200
max_episodes = 10000
total_steps = 0
memory_size = max_steps * max_episodes

memory = ExperienceReplayMemory(memory_size=memory_size)
from SubgoalDiscovery import SubgoalDiscovery
sd = SubgoalDiscovery(n_clusters=4)

epsilon = 0.1
gamma = 0.99
if q_table_is_used:
	lr = 0.01
else:
	lr = 0.0001

FREQ_TEST = 100
REPEAT_TEST = 5
first_fit = False
successful_episodes = []
intrinsic_succesful_episodes = []
n_succesful_task_test = 0
n_succesful_intrinsic_motivation_test = 0
success_rate_intrinsic_motivation_test = []

subgoal_outliers = [] # outliers -- subgoals
subgoals_centroids = [] # center of kmean clusters -- subgoals
G = [] # all subgoals

epsilon_g = 0.001 # probablity of a random subgoal vs greedy

def test_intrinsic_motivation_learning():
	global G
	s = env.reset()
	s0 = copy(s)
	g = env.set_epsilon_greedy_type_subgoal(G, epsilon_g)
	Q = model.compute_Q(s, g)
	a = Q.argmax()
	j = 0
	while j < max_steps:
		sp, r, true_terminal, step_info = env.step(a)
		r_intrinsic, terminal = env.get_intrinsic_reward(sp,g,r)
		if RENDER:
			env.render()
		Qp = model.compute_Qp(sp, g)
		ap = Qp.argmax()
		done_mask = 1 if terminal else 0
		if q_table_is_used:
			delta = r_intrinsic + (1 - done_mask) * gamma * Qp[ap] - Q[a]
			model.update_Q_table(s,g,a,lr*delta)
		else:
			delta = r_intrinsic + (1 - done_mask) * gamma * Qp[0,ap] - Q[0,a]
			model.update_w(a, lr*delta)	

		model.update_Q_to_Qp()
		s = copy(sp)
		a = copy(ap)		
		
		j += 1

		if terminal:
			if VERBOSE:
				print('test of intrinsic motivation --> succesful')
			return True
		return False


def test_original_task():
	global G
	if VERBOSE:
		print('testing original task')
	s = env.reset()
	s0 = copy(s)
	# print('goal: key')
	g = (2,4)
	j = 0
	Q = model.compute_Q(s, g)
	# a = model.epsilon_greedy(Q, epsilon)
	a = Q.argmax()
	while j < max_steps:
		sp, r, true_terminal, step_info = env.step(a)
		r_intrinsic, terminal = env.get_intrinsic_reward(sp,g,r)
		if RENDER:
			env.render()
		Qp = model.compute_Qp(sp, g)
		ap = Qp.argmax()
		done_mask = 1 if terminal else 0
		if q_table_is_used:
			delta = r_intrinsic + (1 - done_mask) * gamma * Qp[ap] - Q[a]
			model.update_Q_table(s,g,a,lr*delta)
		else:
			delta = r_intrinsic + (1 - done_mask) * gamma * Qp[0,ap] - Q[0,a]
			model.update_w(a, lr*delta)	

		model.update_Q_to_Qp()
		s = copy(sp)
		a = copy(ap)		
		
		j += 1

		if true_terminal:
			if VERBOSE:
				print('testing original task succesful!')
			return True
			break

		if terminal:
			if VERBOSE:
				print('testing -- reached to key')
			# print('goal: car')
			g = (11,5)
			j = 0
	return False


for i in range(max_episodes):
	############# Incremental Subgoal Discovery #########################
	# first subgoal discovery
	if i % FREQ_TEST == 0 and i != 0 and first_fit is False:
		X = memory.X
		sd.feed_data(X)
		sd.find_kmeans_clusters(init='random')
		centroids = sd.kmeans.cluster_centers_
		centroids_discrete = np.round_(centroids*16+0.5)
		first_fit = True
		discovered_subgoals_tmp = [ tuple(g) for g in list(centroids_discrete) ]
		subgoals_centroids = discovered_subgoals_tmp
		G = subgoals_centroids + subgoal_outliers

	# next subgoal discovery
	if i % FREQ_TEST == 0 and i != 0 and first_fit is True:
		new_X = np.append(X, memory.X, axis=0)
		sd.feed_data(new_X)
		sd.find_kmeans_clusters(init=centroids)
		centroids = sd.kmeans.cluster_centers_
		centroids_discrete = np.round_(centroids*16+0.5)
		discovered_subgoals_tmp = [ tuple(g) for g in list(centroids_discrete) ]
		X = memory.X
		subgoals_centroids = discovered_subgoals_tmp
		G = subgoals_centroids + subgoal_outliers

	if i % FREQ_TEST == 0 and i != 0:
		test_done = test_original_task()
		if test_done:
			n_succesful_task_test += 1


	if i % FREQ_TEST == 0 and i != 0:
		n_succesful_intrinsic_motivation_test = 0
		for k in range(REPEAT_TEST):
			if VERBOSE:
				print('testing intrinsic motivation learnig: #', k)
			test_intrinsic_motivation = test_intrinsic_motivation_learning()
			if test_intrinsic_motivation:
				n_succesful_intrinsic_motivation_test += 1
		rate = n_succesful_intrinsic_motivation_test / REPEAT_TEST
		success_rate_intrinsic_motivation_test.append(rate)

	if VERBOSE:
		print('-'*60)
		print('episode: ', i)
	s = env.reset()
	s0 = copy(s)

	g = env.set_epsilon_greedy_type_subgoal(G, epsilon_g)
	Q = model.compute_Q(s, g)
	a = model.epsilon_greedy(Q, epsilon)
	j = 0
	while j < max_steps:
		sp, r, true_terminal, step_info = env.step(a)
		r_intrinsic, terminal = env.get_intrinsic_reward(sp,g,r)
		if RENDER:
			env.render()
		# if i == 0:
		# 	env.render()
		total_steps += 1
		experience = (s,a,r,sp)
		memory.push(experience)
		# Outlier -- one class SVM
		if r > 1:
			outlier = sp
			if outlier not in subgoal_outliers:
				subgoal_outliers.append(outlier)
				print('Outlier detected:', outlier)
			if outlier not in G:
				G.append(outlier)
				print(G)

		Qp = model.compute_Qp(sp, g)
		ap = model.epsilon_greedy(Qp, epsilon)
		done_mask = 1 if terminal else 0
		if q_table_is_used:
			delta = r_intrinsic + (1 - done_mask) * gamma * Qp[ap] - Q[a]
			model.update_Q_table(s,g,a,lr*delta)
		else:
			delta = r_intrinsic + (1 - done_mask) * gamma * Qp[0,ap] - Q[0,a]
			model.update_w(a, lr*delta)	

		model.update_Q_to_Qp()
		s = copy(sp)
		a = copy(ap)		
		j += 1

		if true_terminal:
			print('solved the rooms task!')
			successful_episodes.append(i)
			break

		if terminal:
			# print('reached to the intrinsic goal: ', g, 'from', s0)
			intrinsic_succesful_episodes.append(i)
			# print('setting another subgoal')
			g = env.set_epsilon_greedy_type_subgoal(G, epsilon_g)
			j = 0
			Q = model.compute_Q(s, g)
			a = model.epsilon_greedy(Q, epsilon)

env.close()


# rewards = memory.get_rewards()
# x_axis = list(range(len(memory.memory)))

# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 20})
# rc('text', usetex=True)
# rc('xtick', labelsize=18)
# rc('ytick', labelsize=18)
# rc('axes', labelsize=24)
# rc('figure', titlesize=24)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# plt.title('Reward vs. time')
# fig = plt.figure()
# fig.set_size_inches(8, 6)
# plt.plot(x_axis, rewards, '.-')
# plt.xlabel('Time steps $t$')
# plt.ylabel('Reward')
# plot_path = './plots/rooms_rewards_all_episodes.eps'
# plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')


# for j in successful_episodes:
# 	x_axis = list(range(j*max_steps, (j+1)*max_steps))
# 	fig = plt.figure()
# 	fig.set_size_inches(8, 6)
# 	plt.plot(x_axis, rewards[j*max_steps:(j+1)*max_steps], '.-')
# 	plt.xlabel('Time steps $t$')
# 	plt.ylabel('Reward')
# 	plot_path = './plots/rooms_rewards_episode_'+str(j)+'.eps'
# 	plt.savefig(plot_path, format='eps', dpi=1000,bbox_inches='tight')


