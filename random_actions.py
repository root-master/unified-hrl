import time
import gym
from ExperienceReplayMemory import ExperienceReplayMemory
memory = ExperienceReplayMemory(memory_size=10000)
from SubgoalDiscovery import SubgoalDiscovery
sd = SubgoalDiscovery(n_clusters=8)

from gym_rooms.envs import *
env_list = [key for key, _ in gym.envs.registry.env_specs.items()]

environment = 'Rooms-v0'
env = gym.make(environment)
max_steps = 1000
max_episodes = 10
total_steps = 0

for i in range(max_episodes):
	s = env.reset()
	for j in range(max_steps):			
		a = env.action_space.sample()
		sp, r, terminal, step_info = env.step(a)
		env.render()
		total_steps += 1
		experience = (s,a,r,sp)
		memory.push(experience)
		s = sp
		time.sleep(0.05)
		
		if total_steps % 10000 == 0 and total_steps>10:
			sd.feed_data(memory.X)
			sd.find_kmeans_clusters(init='random')
			print(np.round_(sd.kmeans.cluster_centers_[:,0:2]*16))

		if terminal:
			print('solved the rooms task!')
			break


env.close()

# hallway 1:  (8, 13) (7, 13)
# hallway 2:  (8, 2) (7, 2)
# hallway 3:  (3, 7) (3, 8)
# hallway 4:  (14, 7) (14, 8)
# key  :  (2, 4)
# car  :  (11, 5)



