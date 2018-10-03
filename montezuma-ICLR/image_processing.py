import time
import sys
import cv2
import numpy as np
import copy
import sys
import random
import math
from collections import namedtuple
Mask = namedtuple('Mask', 'x y w h')

class Recognizer:
	def __init__(self):
		self.colors = {'man': [200, 72, 72], 'skull': [236,236,236]}
		self.map = {'man': 0, 'ladder': 1, 'rope':2, 'key': 3, 'door': 4}
		self.base_img = cv2.imread('./templates/image.png')
		# self.img_edge = self.base_img[55:181,10:150,:]
		# im2, contours, hierarchy = edge_detector(img_edge)
		self.coords = self.get(self.base_img)
		self.random_subgoals_set = \
			self.create_random_subgoal_set_from_objects()
		# self.discovered_subgoals_set = self.get_discovered_subgoal_set()
		self.discovered_subgoals_set = self.get_discovered_subgoal_set_6()
		self.random_subgoals_set = self.discovered_subgoals_set
		
	def blob_detect(self, img, obj):
		img_man = img[30:,:,:]
		mask = np.zeros(np.shape(img_man))
		mask[:,:,0] = self.colors[obj][0];
		mask[:,:,1] = self.colors[obj][1];
		mask[:,:,2] = self.colors[obj][2];

		diff = img_man - mask
		indxs = np.where(diff == 0)
		diff[np.where(diff < 0)] = 0
		diff[np.where(diff > 0)] = 0
		diff[indxs] = 255
		mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
		mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
		loc = (int(mean_x), int(mean_y+30))
		
		template = cv2.imread('templates/' + obj + '.png')
		w = np.shape(template)[1]
		h = np.shape(template)[0]
		return loc, w, h

	def template_detect(self, img, obj):
		template = cv2.imread('templates/' + obj + '.png')
		w = np.shape(template)[1]
		h = np.shape(template)[0]
		res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
				
		threshold = 0.8

		if obj == 'rope' or obj=='door':
			threshold = 0.95

		loc_np = np.where( res >= threshold)
		# we have to transfer to the image coordinate
		loc = (loc_np[1],loc_np[0])
		loc[0].setflags(write=True)
		loc[1].setflags(write=True)
		return loc, w, h

	def get(self, img):
		man_coords, man_w, man_h = self.blob_detect(img, 'man')
		rope_coords, rope_w, rope_h = self.template_detect(img, 'rope')
		ladder_coords, ladder_w, ladder_h = self.template_detect(img, 'ladder')
		key_coords, key_w, key_h = self.template_detect(img, 'key')
		door_coords, door_w, door_h = self.template_detect(img, 'door')

		dict_result = {	'man': man_coords,
						'man_w': man_w,
						'man_h': man_h,
						'rope': rope_coords,
						'rope_w': rope_w,
						'rope_h': rope_h,
						'ladder':ladder_coords, 
						'key':key_coords, 
						'door':door_coords, 
						'ladder_w': ladder_w,
						'ladder_h':ladder_h	, 
						'key_w':key_w, 
						'key_h':key_h, 
						'door_w':door_w, 
						'door_h':door_h}
		return dict_result

	def drawbbox(self, inputim, coords):
		img = draw_key(inputim,coords)
		img = draw_ladder(img, coords)
		img = draw_door(img, coords)
		img = draw_rope(img, coords)
		return img

	def get_man_mask(self,frames=None):
		if frames is None:
			p, w, h = self.blob_detect(self.base_img,'man')
		else:
			s_max = np.max(np.stack(frames), axis=0)
			p, w, h = self.blob_detect(s_max,'man')
		return Mask(p[0]-w//2, p[1]-h//2, w, h)

	def create_random_subgoal_set_from_objects(self):
		coords = self.coords
		subgoal_set = []
		obj = 'ladder'
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		for loc in zip(*coords[obj]):
			p = ( loc[0], loc[1] ) # top part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			p = ( loc[0], loc[1]+h//2 ) # bottom part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)

		obj = 'key'
		loc = coords[obj]
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		p = ( np.asscalar(loc[0]) , np.asscalar(loc[1]) )
		subgoal_mask = create_mask(p,w,h)
		subgoal_set.append(subgoal_mask)

		obj = 'door'
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		for loc in zip(*coords[obj]):
			p = ( loc[0], loc[1] ) # top part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			p = ( loc[0], loc[1]+h//2 ) # top part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)			
		return subgoal_set

	def get_discovered_subgoal_set(self):
		coords = self.coords
		subgoal_set = []
		self.discovered_subgoal_meaning_set = []
		obj = 'ladder'
		w = coords[obj+'_w']
		h = coords[obj+'_h']

		i = 0		
		for loc in zip(*coords[obj]):
			if i == 0:
				meaning = 'ladder middle: '
			elif i == 1:
				meaning = 'ladder bottom left: '
			elif i == 2:
				meaning = 'ladder bottom right: '
			p = ( loc[0], loc[1]-h//2 ) # stage part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			self.discovered_subgoal_meaning_set.append(meaning+'stage')
			p = ( loc[0], loc[1] ) # top part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			self.discovered_subgoal_meaning_set.append(meaning+'top')
			p = ( loc[0], loc[1]+h//2 ) # bottom part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			self.discovered_subgoal_meaning_set.append(meaning+'bottom')
			i += 1

		obj = 'key'
		loc = coords[obj]
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		p = ( np.asscalar(loc[0]) , np.asscalar(loc[1]) )
		subgoal_mask = create_mask(p,w,h)
		subgoal_set.append(subgoal_mask)
		self.discovered_subgoal_meaning_set.append('key')

		obj = 'door'
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		i = 0
		for loc in zip(*coords[obj]):
			if i == 0:
				meaning = 'left door'
			elif i == 1:
				meaning = 'right door'
			p = ( loc[0], loc[1]+h//2 )
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			self.discovered_subgoal_meaning_set.append(meaning)
			i += 1			
		return subgoal_set

	def get_discovered_subgoal_set_6(self):
		coords = self.coords
		subgoal_set = []
		self.discovered_subgoal_meaning_set = []
		obj = 'ladder'
		w = coords[obj+'_w']
		h = coords[obj+'_h']

		i = 0		
		for loc in zip(*coords[obj]):
			if i == 0:
				meaning = 'ladder middle: '
			elif i == 1:
				meaning = 'ladder bottom left: '
			elif i == 2:
				meaning = 'ladder bottom right: '
			# p = ( loc[0], loc[1]-h//2 ) # stage part
			# subgoal_mask = create_mask(p,w,h//2)
			# subgoal_set.append(subgoal_mask)
			# self.discovered_subgoal_meaning_set.append(meaning+'stage')
			# p = ( loc[0], loc[1] ) # top part
			# subgoal_mask = create_mask(p,w,h//2)
			# subgoal_set.append(subgoal_mask)
			# self.discovered_subgoal_meaning_set.append(meaning+'top')
			p = ( loc[0], loc[1]+h//2 ) # bottom part
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			self.discovered_subgoal_meaning_set.append(meaning+'bottom')
			i += 1

		obj = 'key'
		loc = coords[obj]
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		p = ( np.asscalar(loc[0]) , np.asscalar(loc[1]) )
		subgoal_mask = create_mask(p,w,h)
		subgoal_set.append(subgoal_mask)
		self.discovered_subgoal_meaning_set.append('key')

		obj = 'door'
		w = coords[obj+'_w']
		h = coords[obj+'_h']
		i = 0
		for loc in zip(*coords[obj]):
			if i == 0:
				meaning = 'left door'
			elif i == 1:
				meaning = 'right door'
			p = ( loc[0], loc[1]+h//2 )
			subgoal_mask = create_mask(p,w,h//2)
			subgoal_set.append(subgoal_mask)
			self.discovered_subgoal_meaning_set.append(meaning)
			i += 1			
		return subgoal_set


	def sample_from_random_subgoal_set(self):
		random_subgoal_set = self.random_subgoals_set
		index = random.randint(0, len(random_subgoal_set)-1)
		return index, random_subgoal_set[index]

	def sample_from_discovered_subgoal_set(self):
		discovered_subgoal_set = self.discovered_subgoals_set
		index = random.randint(0, len(discovered_subgoal_set)-1)
		return index, discovered_subgoal_set[index]

	def create_mask_frame(self,mask):
		img = self.base_img
		mask_shape = (img.shape[0],img.shape[1],1)
		mask_frame = np.zeros( mask_shape, dtype=np.uint8)
		xmin = mask.x 
		ymin = mask.y 
		xmax = mask.x + mask.w
		ymax = mask.y + mask.h
		mask_frame[ymin:ymax,xmin:xmax,:] = 255
		return mask_frame

	def edge_detector(self,plotting=True):
		img = copy.deepcopy(self.base_img)
		edges = cv2.Canny(img,100,200)
		im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

		if plotting:
			from matplotlib import pyplot as plt
			plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.title('Original Image'), plt.xticks([]), plt.yticks([])
			plt.subplot(122),plt.imshow(edges,cmap = 'gray')
			plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
			plt.show()
		return im2, contours, hierarchy

	def find_bounding_box(self,plotting=True):
		img = copy.deepcopy(self.base_img)
		edges = cv2.Canny(img,100,200)
		im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
		for c in contours:
			# get the bounding rect
			x, y, w, h = cv2.boundingRect(c)
			# draw a green rectangle to visualize the bounding rect
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

		if plotting:
			from matplotlib import pyplot as plt
			plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.show()
		return im2, contours, hierarchy


def draw_man(inputim,coords):
	obj = 'man'
	img = copy.deepcopy(inputim)
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	pt_0 = (loc[0]-w//2, loc[1]-h//2)
	pt_1 = (loc[0]+w//2, loc[1]+h//2)
	cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img		

def draw_obj(inputim,coords,obj):
	img = copy.deepcopy(inputim)
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj]):
		pt_0 = ( loc[0], loc[1])
		pt_1 = ( loc[0]+w, loc[1]+h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_key(inputim,coords):
	obj = 'key'
	img = copy.deepcopy(inputim)
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	pt_0 = ( loc[0] , loc[1] )
	pt_1 = ( loc[0] + w, loc[1]+h)
	cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_ladder(inputim,coords):
	obj = 'ladder'
	img = copy.deepcopy(inputim)
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj]):
		pt_0 = ( loc[0], loc[1])
		pt_1 = ( loc[0]+w, loc[1]+h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_door(inputim,coords):
	obj = 'door'
	img = copy.deepcopy(inputim)
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj]):
		pt_0 = ( loc[0], loc[1] )
		pt_1 = ( loc[0]+w, loc[1]+h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_rope(inputim,coords):
	obj = 'rope'
	img = copy.deepcopy(inputim)
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj]):
		pt_0 = ( loc[0], loc[1] )
		pt_1 = ( loc[0]+w, loc[1]+h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_subgoal(inputim,g):
	img = copy.deepcopy(inputim)
	pt_0 = (g.x, g.y)
	pt_1 = (g.x+g.w, g.y+g.h)
	cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_circle(inputim,center,radius,color):
	img = copy.deepcopy(inputim)
	cv2.circle(img, center, radius, color, -1)
	return img

def show(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def create_mask(p,w,h):
	return Mask(p[0], p[1], w, h)


def create_random_subgoal_set_from_edges(img,coords):
	'''TODO'''
	pass

def are_masks_align(mask_1, mask_2,threshold=0.8):
	area_1 = mask_1.w * mask_1.h
	area_2 = mask_2.w * mask_2.h
	min_area = min(area_1,area_2)

	xmin_1 = mask_1.x 
	ymin_1 = mask_1.y 
	xmax_1 = mask_1.x + mask_1.w
	ymax_1 = mask_1.y + mask_1.h

	xmin_2 = mask_2.x 
	ymin_2 = mask_2.y 
	xmax_2 = mask_2.x + mask_2.w 
	ymax_2 = mask_2.y + mask_2.h 

	dx = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
	dy = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
	area = abs(dx*dy)
	if (area / min_area) > threshold:
		overlap = True
	else:
		overlap = False
	return overlap

def is_man_inside_subgoal_mask(mask_1, mask_2):
	# 1 --> man, 2--> subgoal
	x_subgoal = mask_2.x + mask_2.w//2
	y_subgoal = mask_2.y + mask_2.h//2

	x_man = mask_1.x + mask_1.w//2
	y_man = mask_1.y + mask_2.h//2

	if ( (x_man - x_subgoal )**2 + (y_man - y_subgoal)**2 <= 64 ):
		return True
	else:
		return False

def get_man_xy_np_coordinate(man_mask):
	x = man_mask.x + man_mask.w//2
	y = man_mask.y + man_mask.h//2
	return np.array([x,y])

def four_frames_to_4_84_84(S):
	""" 0) Atari frames: 210 x 160
		1) Get image grayscale
		2) Rescale image 110 x 84
		3) Crop center 84 x 84 (you can crop top/bottom according to the game)
		"""
	for i, img in enumerate(S):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		h, w = 110, 84
		gray_resized = cv2.resize(gray,(w,h))
		gray_cropped = gray_resized[13:110-13,:]
		gray_reshaped = gray_cropped.reshape((1,84,84))
		if i == 0:
			s = gray_reshaped
		else:
			s = np.concatenate((s,gray_reshaped),axis=0)
	return s

def single_channel_frame_to_1_84_84(subgoal_frame):
	reshaped = subgoal_frame.reshape((210,160))
	resized =  cv2.resize(reshaped,(84,110))
	cropped = resized[13:110-13,:]
	g = cropped.reshape((1,84,84))
	return g






