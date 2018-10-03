# create random subgoals from the edges
# from image_processing import Recognizer
# rec = Recognizer()

# # create the environment
# from Environment import Environment
# task = 'MontezumaRevenge-v0'
# env = Environment(task=task)

# # create expereince memory for Controller
# from memory import ExperienceMemory
# controller_experience_memory = ExperienceMemory() 

# # create subgoal discovery unit
# from subgoal_discovery import SubgoalDiscovery
# sd = SubgoalDiscovery()

# # create Controller 
# from hrl import Controller
# controller = Controller(experience_memory=controller_experience_memory)

# from trainer import IntrinsicMotivation
# intrinsic_motivation_trainer = IntrinsicMotivation( env=env,
# 				 									controller=controller,
# 				 									experience_memory=controller_experience_memory,
# 				 									image_processor=rec,
# 				 									subgoal_discovery=sd)
# intrinsic_motivation_trainer.train()

# phase II:
from image_processing import Recognizer
rec = Recognizer()
G = rec.discovered_subgoals_set
num_options = len(G)

# # create the environment
from Environment import Environment
task = 'MontezumaRevenge-v0'
env = Environment(task=task)

# create expereince memory for Controller
from memory import ExperienceMemory
controller_experience_memory = ExperienceMemory() 

# create expereince memory for Controller
from memory import ExperienceMemoryMeta
meta_controller_experience_memory = ExperienceMemoryMeta() 

# create subgoal discovery unit
from subgoal_discovery import SubgoalDiscovery
sd = SubgoalDiscovery()

# create Controller 
from hrl import Controller
controller = Controller(experience_memory=controller_experience_memory,
						load_pretrained=True,
						saved_model_path='./models/controller_step_2500000.model')

# create Meta-Controller 
from hrl import MetaController
meta_controller = MetaController(meta_controller_experience_memory=meta_controller_experience_memory,
								num_options=num_options)

# create Trainer (Deep HRL Q-Learning loop)
from trainer import MetaControllerController
meta_controller_controller_trainer = \
	MetaControllerController(  
		env=env,
		controller=controller,
		meta_controller=meta_controller,
		experience_memory=controller_experience_memory,
		meta_controller_experience_memory=meta_controller_experience_memory,
		image_processor=rec,
		subgoal_discovery=sd)

meta_controller_controller_trainer.train()
