import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_rooms.rendering import *
from gym.envs.registration import register
import random
import time

CELL_PIXELS = 32 # pixel size

COLORS = {
	'red'   : (255, 0, 0),
	'green' : (0, 255, 0),
	'blue'  : (0, 0, 255),
	'purple': (112, 39, 195),
	'yellow': (255, 255, 0),
	'grey'  : (100, 100, 100),
	'white' : (255,255,255),
	'black' : (0, 0, 0)
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
	'red'   : 0,
	'green' : 1,
	'blue'  : 2,
	'purple': 3,
	'yellow': 4,
	'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
	'empty': 0,
	'wall' : 1,
	'key'  : 2,
	'box'  : 3,
	'hallway':4
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

class WorldObj:
	"""
	Base class for grid world objects
	"""

	def __init__(self, type_obj, color):
		self.type = type_obj
		self.color = color
		self.contains = None

	def render(self, r):
		return

	def _setColor(self, r):
		c = COLORS[self.color]
		r.setLineColor(c[0], c[1], c[2])
		r.setColor(c[0], c[1], c[2])

class Wall(WorldObj):
	def __init__(self, color='grey'):
		super(Wall, self).__init__('wall', color)

	def render(self, r):
		self._setColor(r)
		r.drawPolygon([
			(0          , CELL_PIXELS),
			(CELL_PIXELS, CELL_PIXELS),
			(CELL_PIXELS,           0),
			(0          ,           0)
		])


class HallWay(WorldObj):
	def __init__(self, color='black'):
		super(HallWay, self).__init__('hallway', color)
	
	def render(self,r):
		pass

class Key(WorldObj):
	def __init__(self, color='blue'):
		super(Key, self).__init__('key', color)

	def render(self, r):
		self._setColor(r)

		# Vertical quad
		r.drawPolygon([
			(16, 10),
			(20, 10),
			(20, 28),
			(16, 28)])

		# Teeth
		r.drawPolygon([
			(12, 19),
			(16, 19),
			(16, 21),
			(12, 21)])

		r.drawPolygon([
			(12, 26),
			(16, 26),
			(16, 28),])

		r.drawCircle(18, 9, 6)
		r.setLineColor(0, 0, 0)
		r.setColor(0, 0, 0)
		r.drawCircle(18, 9, 2)

class Box(WorldObj):
	def __init__(self, color, contains=None):
		super(Box, self).__init__('box', color)
		self.contains = contains

	def render(self, r):
		c = COLORS[self.color]
		r.setLineColor(c[0], c[1], c[2])
		r.setColor(0, 0, 0)
		r.setLineWidth(2)

		r.drawPolygon([
			(4            , CELL_PIXELS-4),
			(CELL_PIXELS-4, CELL_PIXELS-4),
			(CELL_PIXELS-4,             4),
			(4            ,             4)
		])

		r.drawLine(
			4,
			CELL_PIXELS / 2,
			CELL_PIXELS - 4,
			CELL_PIXELS / 2
		)

		r.setLineWidth(1)

class Grid:
	"""
	Represent a grid and operations on it
	"""

	def __init__(self, width, height):
		assert width >= 4
		assert height >= 4

		self.width = width
		self.height = height

		self.grid = [None] * width * height

		self.step_info = {'has_key': False, 'has_car': False}
		self.terminal  = False

	def __contains__(self, key):
		if isinstance(key, WorldObj):
			for e in self.grid:
				if e is key:
					return True
		elif isinstance(key, tuple):
			for e in self.grid:
				if e is None:
					continue
				if (e.color, e.type) == key:
					return True
		return False

	def copy(self):
		from copy import deepcopy
		return deepcopy(self)

	def set(self, i, j, v):
		assert i >= 0 and i < self.width
		assert j >= 0 and j < self.height
		self.grid[j * self.width + i] = v

	def get(self, i, j):
		assert i >= 0 and i < self.width
		assert j >= 0 and j < self.height
		return self.grid[j * self.width + i]

	def horzWall(self, x, y, length=None):
		if length is None:
			length = self.width - x
		for i in range(0, length):
			self.set(x + i, y, Wall())

	def vertWall(self, x, y, length=None):
		if length is None:
			length = self.height - y
		for j in range(0, length):
			self.set(x, y + j, Wall())

	def wallRect(self, x, y, w, h):
		self.horzWall(x, y, w)
		self.horzWall(x, y+h-1, w)
		self.vertWall(x, y, h)
		self.vertWall(x+w-1, y, h)

	def render(self, r, tileSize):
		"""
		Render this grid at a given scale
		:param r: target renderer object
		:param tileSize: tile size in pixels
		"""

		assert r.width == self.width * tileSize
		assert r.height == self.height * tileSize

		# Total grid size at native scale
		widthPx = self.width * CELL_PIXELS
		heightPx = self.height * CELL_PIXELS

		# Draw background (out-of-world) tiles the same colors as walls
		# so the agent understands these areas are not reachable
		c = COLORS['white']
		r.setLineColor(c[0], c[1], c[2])
		r.setColor(c[0], c[1], c[2])
		r.drawPolygon([
			(0    , heightPx),
			(widthPx, heightPx),
			(widthPx,      0),
			(0    ,      0)
		])

		r.push()

		# Internally, we draw at the "large" full-grid resolution, but we
		# use the renderer to scale back to the desired size
		r.scale(tileSize / CELL_PIXELS, tileSize / CELL_PIXELS)

		# Draw the background of the in-world cells black
		r.fillRect(
			0,
			0,
			widthPx,
			heightPx,
			255, 255, 255
		)

		# Draw grid lines
		r.setLineColor(100, 100, 100)
		for rowIdx in range(0, self.height):
			y = CELL_PIXELS * rowIdx
			r.drawLine(0, y, widthPx, y)
		for colIdx in range(0, self.width):
			x = CELL_PIXELS * colIdx
			r.drawLine(x, 0, x, heightPx)

		# Render the grid
		for j in range(0, self.height):
			for i in range(0, self.width):
				cell = self.get(i, j)
				if cell is None or cell.type == 'hallway':
					continue
				r.push()
				r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
				cell.render(r)
				r.pop()

		r.pop()

	def encode(self):
		"""
		Produce a compact numpy encoding of the grid
		"""

		codeSize = self.width * self.height * 3

		array = np.zeros(shape=(self.width, self.height, 3), dtype='uint8')

		for j in range(0, self.height):
			for i in range(0, self.width):

				v = self.get(i, j)

				if v == None:
					continue

				array[i, j, 0] = OBJECT_TO_IDX[v.type]
				array[i, j, 1] = COLOR_TO_IDX[v.color]
				array[i, j, 2] = 1

		return array

	def decode(array):
		"""
		Decode an array grid encoding back into a grid
		"""

		width = array.shape[0]
		height = array.shape[1]
		assert array.shape[2] == 3

		grid = Grid(width, height)

		for j in range(0, height):
			for i in range(0, width):

				typeIdx  = array[i, j, 0]
				colorIdx = array[i, j, 1]
				openIdx  = array[i, j, 2]

				if typeIdx == 0:
					continue

				objType = IDX_TO_OBJECT[typeIdx]
				color = IDX_TO_COLOR[colorIdx]
				isOpen = True if openIdx == 1 else 0

				if objType == 'wall':
					v = Wall(color)
				elif objType == 'ball':
					v = Ball(color)
				elif objType == 'key':
					v = Key(color)
				elif objType == 'box':
					v = Box(color)
				elif objType == 'door':
					v = Door(color, isOpen)
				elif objType == 'locked_door':
					v = LockedDoor(color, isOpen)
				elif objType == 'hallway':
					v = HallWay()
				elif objType == 'goal':
					v = Goal()
				else:
					assert False, "unknown obj type in decode '%s'" % objType

				grid.set(i, j, v)

		return grid

class MiniGridEnv(gym.Env):
	"""
	2D grid world game environment
	"""
	metadata = {
		'render.modes': ['human', 'rgb_array', 'pixmap'],
		'video.frames_per_second' : 10
	}

	# Enumeration of possible actions
	class Actions(IntEnum):
		up = 0
		down = 1
		right = 2
		left = 3

	def __init__(self, gridSize=16, maxSteps=1000):
		# Action enumeration for this environment
		self.actions = MiniGridEnv.Actions

		# Actions are discrete integer values
		self.action_space = spaces.Discrete(len(self.actions))

		# Range of possible rewards
		self.reward_range = (-10, 1000)

		# Renderer object used to render the whole grid (full-scale)
		self.gridRender = None

		self.cross_hallway = False

		# # Renderer used to render observations (small-scale agent view)
		# self.obsRender = None

		# Environment configuration
		self.gridSize = gridSize
		self.maxSteps = maxSteps
		self.startPos = (1, 1)
		self.startDir = 0
		self.grid = None
		# Initialize the state
		self.seed()
		self.reset()

	def _genGrid(self, width, height):
		assert False, "_genGrid needs to be implemented by each environment"

	def reset(self):
		# Generate a new random grid at the start of each episode
		# To keep the same grid for each episode, call env.seed() with
		# the same seed before calling env.reset()
		self._genGrid(self.gridSize, self.gridSize)

		# Place the agent in the starting position and direction
		self.agentPos = self.startPos
		self.agentDir = self.startDir
		obs = self.agentPos
		state = (obs[0],self.gridSize-obs[1]-1)
		self.state = state
		return state

	def seed(self, seed=1337):
		# Seed the random number generator
		# seed = random.randint(0,400000)
		# seed = int(time.time())
		self.np_random, _ = seeding.np_random(seed)

		return [seed]

	def _randInt(self, low, high):
		"""
		Generate random integer in [low,high[
		"""
		return self.np_random.randint(low, high)

	def _randElem(self, iterable):
		"""
		Pick a random element in a list
		"""
		lst = list(iterable)
		idx = self._randInt(0, len(lst))
		return lst[idx]

	def _randPos(self, xLow, xHigh, yLow, yHigh):
		"""
		Generate a random (x,y) position tuple
		"""
		return (
			self.np_random.randint(xLow, xHigh),
			self.np_random.randint(yLow, yHigh)
		)

	def placeObj(self, obj):
		"""
		Place an object at an empty position in the grid
		"""
		while True:
			pos = (
				self._randInt(0, self.grid.width),
				self._randInt(0, self.grid.height)
			)
			if self.grid.get(*pos) != None:
				continue
			if pos == self.startPos:
				continue
			break
		self.grid.set(*pos, obj)
		return pos

	def placeAgent(self, randDir=True):
		"""
		Set the agent's starting point at an empty position in the grid
		"""
		pos = self.placeObj(None)
		self.startPos = pos

		if randDir:
			self.startDir = self._randInt(0, 4)

		return pos

	def getDirVec(self):
		# our coordinate system vs. Qimage
		# ^				   	------->
		# |			  		|
		# |			  		|
		#  ----->     		v
		if self.agentDir == 0: # up (down in image plane)
			return (0, -1)

		elif self.agentDir == 1: # down (up in image plane)
			return (0, 1)

		elif self.agentDir == 2: # right
			return (1, 0)

		elif self.agentDir == 3: # left
			return (-1, 0)
		else:
			assert False

	def step(self, action):
		reward = 0
		self.agentDir = action
		terminal = False
		(u, v) = self.getDirVec()
		oldPos = self.agentPos
		newPos = (self.agentPos[0] + u, self.agentPos[1] + v)

		targetCell = self.grid.get(newPos[0], newPos[1])
		if targetCell == None:
			self.agentPos = newPos
		elif targetCell.type == 'wall':
			reward = -2 # bumped to wall
			newPos = oldPos
		elif targetCell.type == 'key':
			reward = +10
			if self.step_info['has_key'] == False:
				self.step_info['has_key'] = True
				self.grid.set(newPos[0], newPos[1], None)
		elif targetCell.type == 'box' and self.step_info['has_key'] == True:
			reward = +40
			self.step_info['has_car'] = True
			terminal = True

		self.state_before_passing_doorway = (newPos[0],self.gridSize-newPos[1]-1)
		######### PASS THE HALLWAY ###########
		if newPos in self.hallways and self.cross_hallway is True:
			u = u * 3
			v = v * 3
			newPos = (self.agentPos[0] + u, self.agentPos[1] + v)

		self.agentPos = newPos
		state = (newPos[0],self.gridSize-newPos[1]-1)
		self.state = state

		return state, reward, terminal, self.step_info

	def render(self, mode='human', close=False):
		"""
		Render the whole-grid human view
		"""
		if close:
			if self.gridRender:
				self.gridRender.close()
			return

		if self.gridRender is None:
			self.gridRender = Renderer(
				self.gridSize * CELL_PIXELS,
				self.gridSize * CELL_PIXELS,
				True if mode == 'human' else False
			)

		r = self.gridRender

		r.beginFrame()

		# Render the whole grid
		self.grid.render(r, CELL_PIXELS)

		# Draw the agent
		r.push()
		r.translate(
			CELL_PIXELS * (self.agentPos[0] + 0.5),
			CELL_PIXELS * (self.agentPos[1] + 0.5)
		)
		if self.agentDir == 0:
			angle = 270
		elif self.agentDir == 1:
			angle = 90
		elif self.agentDir == 2:
			angle = 0
		elif self.agentDir == 3:
			angle = 180
		r.rotate(angle)
		r.setLineColor(255, 0, 0)
		r.setColor(255, 0, 0)
		r.drawPolygon([
			(-12, 10),
			( 12,  0),
			(-12, -10)
		])
		r.pop()

		r.endFrame()

		if mode == 'rgb_array':
			return r.getArray()
		elif mode == 'pixmap':
			return r.getPixmap()

		return r

class Room:
	def __init__(self,top,size,color,objects):		
		self.top = top	
		self.size = size # size of the room
		self.color = color # Color of the room
		self.objects = objects # List of objects contained


class RoomsEnv(MiniGridEnv):
	metadata = {'render.modes': ['human', 'rgb_array', 'pixmap'],
				'video.frames_per_second' : 30}

	class Actions(IntEnum):
		up = 0
		down = 1
		right = 2
		left = 3

	def __init__(self, size=16):
		assert size >= 10
		super(RoomsEnv, self).__init__(gridSize=size, maxSteps=8*size)

		self.actions = RoomsEnv.Actions

		self.action_space = spaces.Discrete(len(self.actions))

		self.reward_range = (-10, 1000)
		self.step_info = {'has_key': False, 'has_car': False}
		self.first_time_reset = True

	def _randPos(self, room, border=1):
		return (
			self._randInt(
				room.top[0] + border,
				room.top[0] + room.size[0] - border
			),
			self._randInt(
				room.top[1] + border,
				room.top[1] + room.size[1] - border
			),
		)

	def _genGrid(self, width, height):
		
		if self.grid is not None:
			self.startDir = self._randInt(0, 4)
			room = self._randElem(self.rooms)
			self.startPos = self._randPos(room)
			for i in range(2):
				obj = self.objects[i]
				objType = self.objType[i]
				pos = self.obj_pos[objType]

				if self.grid.get(*pos) is None:
					self.grid.set(*pos,obj)
				self.step_info = {'has_key': False, 'has_car': False}
			return

		self.grid = Grid(width, height)

		# Horizontal and vertical split indices
		vSplitIdx = self._randInt(5, width-4)
		hSplitIdx = self._randInt(5, height-4)

		# Create the four rooms
		self.rooms = []
		self.rooms.append(Room(
			(0, 0),
			(vSplitIdx, hSplitIdx),
			'red',
			[]
		))
		self.rooms.append(Room(
			(vSplitIdx, 0),
			(width - vSplitIdx, hSplitIdx),
			'purple',
			[]
		))
		self.rooms.append(Room(
			(0, hSplitIdx),
			(vSplitIdx, height - hSplitIdx),
			'blue',
			[]
		))
		self.rooms.append(Room(
			(vSplitIdx, hSplitIdx),
			(width - vSplitIdx, height - hSplitIdx),
			'yellow',
			[]
		))

		# Place the room walls
		for room in self.rooms:
			x, y = room.top
			w, h = room.size

			# Horizontal walls
			for i in range(w):
				self.grid.set(x + i, y, Wall(room.color))
				self.grid.set(x + i, y + h - 1, Wall(room.color))

			# Vertical walls
			for j in range(h):
				self.grid.set(x, y + j, Wall(room.color))
				self.grid.set(x + w - 1, y + j, Wall(room.color))

		# Place wall openings connecting the rooms
		hIdx = self._randInt(1, hSplitIdx-1)
		self.grid.set(vSplitIdx, hIdx, HallWay())
		self.grid.set(vSplitIdx-1, hIdx, HallWay())
		self.hallways = [(vSplitIdx,height-1-hIdx),(vSplitIdx-1,height-1-hIdx)]
		print('hallway 1: ',(vSplitIdx,height-1-hIdx),(vSplitIdx-1,height-1-hIdx))
			
		hIdx = self._randInt(hSplitIdx+1, height-1)
		self.grid.set(vSplitIdx, hIdx, HallWay())
		self.grid.set(vSplitIdx-1, hIdx, HallWay())
		self.hallways.append((vSplitIdx,height-1-hIdx))
		self.hallways.append((vSplitIdx-1,height-1-hIdx))
		print('hallway 2: ',(vSplitIdx,height-1-hIdx),(vSplitIdx-1,height-1-hIdx))
		
		vIdx = self._randInt(1, vSplitIdx-1)
		self.grid.set(vIdx, hSplitIdx, HallWay())
		self.grid.set(vIdx, hSplitIdx-1, HallWay())
		self.hallways.append((vIdx,height-hSplitIdx-1))
		self.hallways.append((vIdx,height-hSplitIdx))
		print('hallway 3: ',(vIdx,height-hSplitIdx-1),(vIdx,height-hSplitIdx))
		vIdx = self._randInt(vSplitIdx+1, width-1)
		self.grid.set(vIdx, hSplitIdx, HallWay())
		self.grid.set(vIdx, hSplitIdx-1, HallWay())
		self.hallways.append((vIdx,height-hSplitIdx-1))
		self.hallways.append((vIdx,height-hSplitIdx))
		print('hallway 4: ',(vIdx,height-hSplitIdx-1),(vIdx,height-hSplitIdx))


		# Select a random position for the agent to start at
		room = self._randElem(self.rooms)
		while True:		
			self.startDir = self._randInt(0, 4)
			pos = self._randPos(room)
			if self.grid.get(*pos) != None:
				continue
			else:
				self.startPos = pos
				break

		self.objType = ['key','car']
		self.objects = [Key('green'), Box('red')]
		self.obj_pos = {'key': (2,2), 'door': (10,10)} # init- gonna change
		for i in range(0, 2):
			obj = self.objects[i]
			# Pick a random position that doesn't overlap with anything
			while True:
				room = self._randElem(self.rooms)
				pos = self._randPos(room, border=2)
				if pos == self.startPos:
					continue
				if self.grid.get(*pos) != None:
					continue
				self.grid.set(*pos, obj)
				objType = self.objType[i]
				self.obj_pos[objType] = pos
				print(self.objType[i],' : ', (pos[0],self.gridSize-pos[1]-1))
				break

			room.objects.append(obj)

	def step(self, action):
		obs, reward, done, info = MiniGridEnv.step(self, action)
		return obs, reward, done, info

	def set_epsilon_greedy_type_subgoal(self, G, epsilon):
		if random.random() < epsilon:
			return self.set_random_intrinsic_goal_from_states()
		else:
			return self.set_random_intrinsic_goal_from_subgoals(G)

	def set_random_intrinsic_goal_from_states(self):
		while True:
			room = self._randElem(self.rooms)
			pos = self._randPos(room, border=2)
			if pos == self.startPos or self.grid.get(*pos) != None:
				continue
			else:
				self.intrinsic_goal_pos = pos
				goal_state = (pos[0],self.gridSize-pos[1]-1)
				self.goal_state = goal_state
				return goal_state

	def set_random_intrinsic_goal_from_subgoals(self, G):
		if len(G) == 0:
			return self.set_random_intrinsic_goal_from_states()
		return self._randElem(G)

	def get_intrinsic_reward(self, s, g, reward):
		if s == g:
			reward = 1
			done = True
		elif reward >= 0:
			reward = -0.1
			done = False
		elif reward<0:
			reward = -0.5
			done = False
		return reward, done 

register(
	id='Rooms-v0',
	entry_point='gym_rooms.envs:RoomsEnv')

