from ExperienceReplayMemory import ExperienceReplayMemory
experience_memory = ExperienceReplayMemory(memory_size=10000)

from SubgoalDiscovery import SubgoalDiscovery
subgoal_discovery = SubgoalDiscovery(n_clusters=8,experience_memory=experience_memory)

import gym
from gym_rooms.envs import *
environment = 'Rooms-v0'
env = gym.make(environment)
 
from trainer import RandomWalk
random_walk = RandomWalk(env=env,subgoal_discovery=subgoal_discovery,experience_memory=experience_memory)

# lets random walk and find the subgoals such as centroids and outliers
random_walk.walk()

# let's do another random walk and find the doorways
random_walk.walk_and_find_doorways()

outliers = subgoal_discovery.outliers
list_outliers = [[g] for g in subgoal_discovery.outliers]
subgoals = subgoal_discovery.doorway_pairs + list_outliers

print(subgoal_discovery.centroid_subgoals)


# K=8
# [(14, 10), (12, 6), (4, 4), (4, 12), (10, 12), (13, 14), (14, 3), (10, 3)]
# [(4, 4), (2, 12), (13, 10), (13, 4), (13, 14), (6, 12), (10, 4), (10, 11)]

# K=6
# [(4, 6), (12, 4), (12, 12), (3, 12), (6, 12), (4, 3)]