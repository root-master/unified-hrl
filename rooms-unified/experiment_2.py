import time
start_time = time.time()

from ExperienceReplayMemory import ExperienceReplayMemory
experience_memory = ExperienceReplayMemory(memory_size=10000)

from SubgoalDiscovery import SubgoalDiscovery
subgoal_discovery = SubgoalDiscovery(n_clusters=6,experience_memory=experience_memory)

import gym
from gym_rooms.envs import *
environment = 'Rooms-v0'
env = gym.make(environment)
 
from trainer import RandomWalk
random_walk = RandomWalk(env=env,subgoal_discovery=subgoal_discovery,experience_memory=experience_memory)

# lets random walk and find the subgoals such as centroids and outliers
random_walk.walk()

outliers = subgoal_discovery.outliers
centroids = subgoal_discovery.centroid_subgoals
subgoals = subgoal_discovery.G

randomwalk_USD_time = time.time()
print('Elapse time for unsupervised subgoal discovery: ', randomwalk_USD_time-start_time)

from hrl import Controller
controller = Controller(subgoal_discovery=subgoal_discovery)

env.cross_hallway = True
from hrl import MetaController
meta_controller = MetaController(subgoal_discovery=subgoal_discovery)

from trainer import MetaControllerControllerUnified
meta_controller_controller_trainer = MetaControllerControllerUnified( env=env,
									controller=controller,
									meta_controller=meta_controller,
									subgoal_discovery=subgoal_discovery,
									pretrain=False)

meta_controller_controller_trainer.train()
meta_controller_controller_time = time.time() - start_time
print('Elapsed time for training meta controller: ', meta_controller_controller_time)

