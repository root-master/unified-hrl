import time
import gym
import numpy as np

from Model import Model_1

from ExperienceReplayMemory import ExperienceReplayMemory

from Model import Model_1
model = Model_1()

from gym_rooms.envs import *
environment = 'Rooms-v0'
env = gym.make(environment)
RENDER = False

max_steps = 200
max_episodes = 100
total_steps = 0
memory_size = max_steps * max_episodes

memory = ExperienceReplayMemory(memory_size=memory_size)
from SubgoalDiscovery import SubgoalDiscovery, UnsupervisedOutlierDetection
sd = SubgoalDiscovery(n_clusters=4)
outlier_detector = UnsupervisedOutlierDetection()

first_fit = False
subgoal_outliers = [] # outliers -- subgoals
subgoals_centroids = [] # center of kmean clusters -- subgoals
G = [] # all subgoals

successful_episodes = []
for i in range(max_episodes):

	# if total_steps % memory_size == 0 and i != 0 and first_fit is False:
	# 	X = memory.X
	# 	sd.feed_data(X)
	# 	sd.find_kmeans_clusters(init='random')
	# 	centroids = sd.kmeans.cluster_centers_
	# 	first_fit = True

	# if total_steps % memory_size == 0 and i != 0 and first_fit is True:
	# 	new_X = np.append(X, memory.X, axis=0)
	# 	sd.feed_data(new_X)
	# 	sd.find_kmeans_clusters(init=centroids)
	# 	centroids = sd.kmeans.cluster_centers_
	# 	X = memory.X
	
	if i == 1:
		outlier_detector.fit_data(memory.get_reward_np())

	s = env.reset()
	for j in range(max_steps):			
		a = env.action_space.sample()
		sp, r, terminal, step_info = env.step(a)

		if i != 0:
			if outlier_detector.detect_outlier(r):
				outlier = sp
				if outlier not in subgoal_outliers:
					subgoal_outliers.append(outlier)
					print('Outlier detected:', outlier)
				if outlier not in G:
					G.append(outlier)
					print(G)
				# pop the outlier out of data
				memory.pop()


		# if r > 1:
		# 	outlier = sp
		# 	if outlier not in subgoal_outliers:
		# 		subgoal_outliers.append(outlier)
		# 		print('Outlier detected:', outlier)
		# 	if outlier not in G:
		# 		G.append(outlier)
		# 		print(G)

		if i == 0 and RENDER:
			env.render()
		total_steps += 1
		experience = (s,a,r,sp)
		memory.push(experience)
		s = sp		
		if terminal:
			print('solved the rooms task!')
			successful_episodes.append(i)
			break

env.close()

memory.get_experience_X_np()
sd.feed_data(memory.X)
sd.find_kmeans_clusters(init='random')
centroids_discrete = np.round_(sd.kmeans.cluster_centers_*16+0.5)
discovered_subgoals_tmp = [ tuple(g) for g in list(centroids_discrete) ]
subgoals_centroids = discovered_subgoals_tmp
G = subgoals_centroids + subgoal_outliers
print(G)

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

# # plt.title('Reward vs. time')
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


