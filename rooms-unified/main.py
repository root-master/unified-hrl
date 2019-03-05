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

outliers = subgoal_discovery.outliers
centroids = subgoal_discovery.centroid_subgoals
subgoals = subgoal_discovery.G

from hrl import Controller
controller = Controller(subgoal_discovery=subgoal_discovery)

env.cross_hallway = True
from trainer import PretrainController
pretainer = PretrainController( env=env,
 								controller=controller,
 								subgoal_discovery=subgoal_discovery)
pretainer.train()
# pretainer.controller.Q.save_model()

# pretainer.controller.Q.load_model()
from hrl import MetaController
meta_controller = MetaController(subgoal_discovery=subgoal_discovery)

# from trainer import MetaControllerController
# meta_controller_trainer = MetaControllerController( env=env,
# 								controller=controller,
# 								meta_controller=meta_controller,
# 								subgoal_discovery=subgoal_discovery)

# meta_controller_trainer.train()

from trainer import MetaControllerControllerUnified
meta_controller_controller_trainer = MetaControllerControllerUnified( env=env,
									controller=pretainer.controller,
									meta_controller=meta_controller,
									subgoal_discovery=subgoal_discovery)

meta_controller_controller_trainer.train()


# from trainer import VanillaRL
# vanilla_rl = VanillaRL(env=env)
# vanilla_rl.train()

