from ExperienceReplayMemory import ExperienceReplayMemory
experience_memory = ExperienceReplayMemory(memory_size=10000)

from SubgoalDiscovery import SubgoalDiscovery
subgoal_discovery = SubgoalDiscovery(n_clusters=4,experience_memory=experience_memory)

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

from hrl import Controller
controller = Controller()

env.cross_hallway = True
from trainer import PretrainController
pretainer = PretrainController( env=env,
								controller=controller,
								subgoals=subgoals)
pretainer.train()
pretainer.controller.Q.save_model()

pretainer.controller.Q.load_model()
from hrl import MetaController
meta_controller = MetaController(subgoal_discovery=subgoal_discovery,
								outliers=outliers)

from trainer import MetaControllerController
meta_controller_trainer = MetaControllerController( env=env,
								controller=controller,
								meta_controller=meta_controller,
								subgoals=subgoals)

meta_controller_trainer.train()

from trainer import VanillaRL
vanilla_rl = VanillaRL(env=env)
vanilla_rl.train()

