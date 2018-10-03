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

	def blob_detect(self, img, obj):
		mask = np.zeros(np.shape(img))
		mask[:,:,0] = self.colors[obj][0];
		mask[:,:,1] = self.colors[obj][1];
		mask[:,:,2] = self.colors[obj][2];

		diff = img - mask
		indxs = np.where(diff == 0)
		diff[np.where(diff < 0)] = 0
		diff[np.where(diff > 0)] = 0
		diff[indxs] = 255
		# the most bottom pixel of the man
		y_id = indxs[0].size
		y = indxs[0][-1]
		x = indxs[1][y_id-1]

		loc = (x, y) 
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

		if obj == 'rope':
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

	def get_man_mask(self,img):
		p, w, h = self.blob_detect(img,'man')
		return Mask(p[0], p[1], w, h)

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

def draw_key(inputim,coords):
	obj = 'key'
	img = copy.deepcopy(inputim)
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	pt_0 = ( loc[0]-w//2 , loc[1] )
	pt_1 = ( loc[0]+3*w//2, loc[1]+h)
	cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_ladder(inputim,coords):
	obj = 'ladder'
	img = copy.deepcopy(inputim)
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj][::-1]):
		pt_0 = ( loc[1], loc[0])
		pt_1 = ( loc[1]+w, loc[0]+h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_door(inputim,coords):
	obj = 'door'
	img = copy.deepcopy(inputim)
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj][::-1]):
		pt_0 = ( loc[1], loc[0] )
		pt_1 = ( loc[1] + w,  loc[0] + h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def draw_rope(inputim,coords):
	obj = 'rope'
	img = copy.deepcopy(inputim)
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj][::-1]):
		pt_0 = ( loc[1], loc[0] )
		pt_1 = ( loc[1] + w,  loc[0] + h)
		cv2.rectangle(img, pt_0, pt_1, (0,0,255), 2)
	return img

def show(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def edge_detector(img,plotting=False):
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

def create_mask(p,w,h):
	return Mask(p[0], p[1], w, h)

def create_random_subgoal_set_from_objects(coords):
	subgoal_set = []
	obj = 'ladder' # bottum  ladder
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']

	for loc in zip(*coords[obj][::-1]):
		p = ( loc[1]+w//2, loc[0]-h//4 ) # top
		subgoal_mask = create_mask(p,w,h//2)
		subgoal_set.append(subgoal_mask)
		p = ( loc[1]+w//2, loc[0]+h//4 ) # middle
		subgoal_mask = create_mask(p,w,h//2)
		subgoal_set.append(subgoal_mask)
		p = ( loc[1]+w//2, loc[0]+3*h//4 ) # bottom
		subgoal_mask = create_mask(p,w,h//2)
		subgoal_set.append(subgoal_mask)


	obj = 'key'
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	p = ( np.asscalar(loc[0])+w//2 , np.asscalar(loc[1])+h//2 )
	subgoal_mask = create_mask(p,2*w,h)
	subgoal_set.append(subgoal_mask)
	obj = 'rope'
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj][::-1]):
		p = ( loc[1]+w//2, loc[0]+h//2 )
		subgoal_mask = create_mask(p,w,h)
		subgoal_set.append(subgoal_mask)
	obj = 'door'
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	for loc in zip(*coords[obj][::-1]):
		p = ( loc[1]+w//2, loc[0]+3*h//4 )
		subgoal_mask = create_mask(p,w,h//2)
		subgoal_set.append(subgoal_mask)	
	return subgoal_set

def create_good_subgoal_set_from_objects(coords):
	subgoal_set = []
	obj = 'ladder'
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	loc = (128, 135) # right ladder
	p = ( loc[0]+w//2, loc[1]+h//4 )
	subgoal_mask = create_mask(p,w,h//2)
	subgoal_set.append(subgoal_mask)

	obj = 'ladder'
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	loc = (16,135) # left ladder
	p = ( loc[0]+w//2, loc[1]+h//4 )
	subgoal_mask = create_mask(p,w,h//2)
	subgoal_set.append(subgoal_mask)

	obj = 'key'
	loc = coords[obj]
	w = coords[obj+'_w']
	h = coords[obj+'_h']
	p = ( np.asscalar(loc[0])+w//2 , np.asscalar(loc[1])+h//2 )
	subgoal_mask = create_mask(p,2*w,h)
	subgoal_set.append(subgoal_mask)

	obj = 'door'
	loc = (16,44) # top left door
	w = coords[obj+'_w']
	h = coords[obj+'_h']	
	p = ( loc[0]+w//2, loc[1]+3*h//4 )
	subgoal_mask = create_mask(p,w,h//2)
	subgoal_set.append(subgoal_mask)	

	obj = 'door'
	loc = (132,44) # top right door
	w = coords[obj+'_w']
	h = coords[obj+'_h']	
	p = ( loc[0]+w//2, loc[1]+3*h//4 )
	subgoal_mask = create_mask(p,w,h//2)
	subgoal_set.append(subgoal_mask)
	
	return subgoal_set


def create_random_subgoal_set_from_edges(img,coords):
	'''TODO'''
	pass

def create_mask_frame(img,mask):
	mask_shape = (img.shape[0],img.shape[1],1)
	mask_frame = np.zeros( mask_shape )
	xmin = mask.x - mask.w // 2
	ymin = mask.y - mask.h // 2
	xmax = mask.x + mask.w // 2
	ymax = mask.y + mask.h // 2
	mask_frame[ymin:ymax,xmin:xmax,:] = 255
	return mask_frame

def are_masks_align(mask_1, mask_2,threshold=0.8):
	area_1 = mask_1.w * mask_1.h
	area_2 = mask_2.w * mask_2.h
	min_area = min(area_1,area_2)

	xmin_1 = mask_1.x - mask_1.w // 2
	ymin_1 = mask_1.y - mask_1.h // 2
	xmax_1 = mask_1.x + mask_1.w // 2
	ymax_1 = mask_1.y + mask_1.h // 2

	xmin_2 = mask_2.x - mask_2.w // 2
	ymin_2 = mask_2.y - mask_2.h // 2
	xmax_2 = mask_2.x + mask_2.w // 2
	ymax_2 = mask_2.y + mask_2.h // 2

	dx = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
	dy = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
	area = abs(dx*dy)
	if (area / min_area) > threshold:
		overlap = True
	else:
		overlap = False
	return overlap

# def is_man_inside_subgoal_mask(mask_1, mask_2):
# 	# 1 --> man, 2--> subgoal
# 	xmin_1 = mask_1.x - mask_1.w // 2
# 	ymin_1 = mask_1.y - mask_1.h // 2
# 	xmax_1 = mask_1.x + mask_1.w // 2
# 	ymax_1 = mask_1.y + mask_1.h // 2

# 	xmin_2 = mask_2.x - mask_2.w // 2
# 	ymin_2 = mask_2.y - mask_2.h // 2
# 	xmax_2 = mask_2.x + mask_2.w // 2
# 	ymax_2 = mask_2.y + mask_2.h // 2

# 	x_man = mask_1.x
# 	y_man = mask_1.y

# 	if (xmin_2<x_man<xmax_2) and (ymin_2<y_man<ymax_2):
# 		return True
# 	else:
# 		return False


def is_man_inside_subgoal_mask(mask_1, mask_2):
	# 1 --> man, 2--> subgoal
	x_subgoal = mask_2.x
	y_subgoal = mask_2.y

	x_man = mask_1.x
	y_man = mask_1.y

	if ( (x_man - x_subgoal )**2 + (y_man - y_subgoal)**2 <= 64 ):
		return True
	else:
		return False




