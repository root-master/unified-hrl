import numpy as np
from image_processing import *
rec = Recognizer()
from copy import copy
img = cv2.imread('./templates/base.png')
# img_edge = img_original[55:181,10:150,:]
# im2, contours, hierarchy = eadge_detector(img_edge)

# import gym
# env = gym.make('MontezumaRevenge-v0')

# img = env.reset()
# bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

coords = rec.get(img)
img = rec.drawbbox(img, coords)
show(img)


# img, r, done, step_info = env.step(env.action_space.sample())
# env.render()
# loc,w,h = rec.blob_detect(rgb_img, 'man')

w = {}
i = 0
for param in Qt.parameters():
	w[i] = copy(param.data)
	i += 1

w_1 = {}
i = 0
for param in Qt.parameters():
	w_1[i] = copy(param.data)
	i += 1
