# create random subgoals from the edges
from image_processing import Recognizer
rec = Recognizer()

# create the environment
from Environment import Environment
task = 'MontezumaRevenge-v0'
env = Environment(task=task)

# create expereince memory for Controller
from memory import ExperienceMemory
controller_experience_memory = ExperienceMemory() 

# create subgoal discovery unit
from subgoal_discovery import SubgoalDiscovery
sd = SubgoalDiscovery()

# create Controller 
from hrl import Controller
controller = Controller(experience_memory=controller_experience_memory)

from trainer import IntrinsicMotivation
intrinsic_motivation_trainer = IntrinsicMotivation( env=env,
				 									controller=controller,
				 									experience_memory=controller_experience_memory,
				 									image_processor=rec,
				 									subgoal_discory=sd)
intrinsic_motivation_trainer.train()



