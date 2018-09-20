import gym 
import time


environment = 'MontezumaRevengeNoFrameskip-v0'
env = gym.make(environment)
actions_meaning = env.unwrapped.get_action_meanings()

game_score = 0.0
env.reset()
for i in range(10000):
	env.render()
	time.sleep(0.01)
	old_lives = env.unwrapped.ale.lives()
	s,r,done,info = env.step(env.action_space.sample())
	game_score += r
	current_lives = env.unwrapped.ale.lives()
	if current_lives < old_lives:
		print('died, lives = ',current_lives)

	if done:
		print('termiated, lives = ', current_lives)
		print('score = ', game_score)
		break

# import numpy as np
# from image_processing import *
# rec = Recognizer()
# from copy import copy
# img = cv2.imread('./templates/base.png')
# # img_edge = img_original[55:181,10:150,:]
# # im2, contours, hierarchy = eadge_detector(img_edge)

# # import gym
# # env = gym.make('MontezumaRevenge-v0')

# # img = env.reset()
# # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# coords = rec.get(img)
# img = rec.drawbbox(img, coords)
# show(img)


# # img, r, done, step_info = env.step(env.action_space.sample())
# # env.render()
# # loc,w,h = rec.blob_detect(rgb_img, 'man')

# w = {}
# i = 0
# for param in Qt.parameters():
# 	w[i] = copy(param.data)
# 	i += 1

# w_1 = {}
# i = 0
# for param in Qt.parameters():
# 	w_1[i] = copy(param.data)
# 	i += 1
