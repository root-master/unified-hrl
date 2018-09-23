# import gym 
# import time

import pickle
t = 2500000
results_file_path = './results/performance_results_' + str(t) + '.pkl'
with open(results_file_path,'rb') as f:
	A=pickle.load(f)

# environment = 'MontezumaRevengeNoFrameskip-v0'
# env = gym.make(environment)
# actions_meaning = env.unwrapped.get_action_meanings()

# game_score = 0.0
# env.reset()
# for i in range(10000):
# 	env.render()
# 	time.sleep(0.01)
# 	old_lives = env.unwrapped.ale.lives()
# 	s,r,done,info = env.step(env.action_space.sample())
# 	game_score += r
# 	current_lives = env.unwrapped.ale.lives()
# 	if current_lives < old_lives:
# 		print('died, lives = ',current_lives)

# 	if done:
# 		print('termiated, lives = ', current_lives)
# 		print('score = ', game_score)
# 		break

# import numpy as np
from image_processing import *
rec = Recognizer()
# C = [(124.0, 167.0),
#  (127.0, 82.0),
#  (64.0, 128.0),
#  (143.0, 127.0),
#  (111.0, 117.0),
#  (30.0, 82.0),
#  (78.0, 83.0),
#  (35.0, 171.0),
#  (79.0, 103.0),
#  (88.0, 126.0)]

# C = [(124.0, 167.0),
#  (127.0, 82.0),
#  (64.0, 128.0),
#  (143.0, 127.0),
#  (111.0, 117.0),
#  (30.0, 82.0),
#  (78.0, 83.0),
#  (49.0, 171.0),
#  (79.0, 103.0),
#  (88.0, 126.0)]

# C = [(123.0, 166.0),
#  (123.0, 81.0),
#  (63.0, 128.0),
#  (139.0, 128.0),
#  (110.0, 117.0),
#  (33.0, 80.0),
#  (78.0, 80.0),
#  (47.0, 171.0),
#  (79.0, 102.0),
#  (87.0, 125.0)]

# C = [(120.0, 167.0),
#  (117.0, 81.0),
#  (62.0, 127.0),
#  (136.0, 128.0),
#  (110.0, 118.0),
#  (43.0, 82.0),
#  (79.0, 79.0),
#  (46.0, 171.0),
#  (80.0, 102.0),
#  (88.0, 126.0)]

# C = [(121.0, 167.0),
#  (121.0, 80.0),
#  (62.0, 127.0),
#  (137.0, 128.0),
#  (110.0, 117.0),
#  (36.0, 79.0),
#  (78.0, 79.0),
#  (46.0, 171.0),
#  (79.0, 102.0),
#  (88.0, 125.0)]

# C = [(125.0, 166.0),
#  (126.0, 82.0),
#  (64.0, 128.0),
#  (142.0, 127.0),
#  (111.0, 118.0),
#  (31.0, 81.0),
#  (78.0, 82.0),
#  (48.0, 171.0),
#  (79.0, 103.0),
#  (88.0, 126.0)]

C = [(79, 101),
 (83, 120),
 (65, 126),
 (128, 159),
 (110, 118),
 (136, 127),
 (78, 81),
 (120, 81),
 (44, 170),
 (33, 82)]

O = [(12,120),
	(14,116),
	(19,116),
	(29,84),
	(127,83)]

G = C + O

img = rec.base_img
color = (0,0,255)
for g in C:
	g = (int(g[0]),int(g[1]))
	img = draw_circle(img, g, 4, color)

color = (255,100,0)
for g in O:
	g = (int(g[0]),int(g[1]))
	img = draw_circle(img, g, 4, color)

# show(img)
from matplotlib import pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


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
