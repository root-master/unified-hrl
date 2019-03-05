import numpy as np
from copy import copy 
from random import random, randint
import pickle

def normpdf(x_vec,mu,sigma):
	# normal probability distribution with mean mu and std sigma
	x = np.exp(- np.square(x_vec - mu) / (2 * sigma * sigma))
	return x

def input_matrix(s):
	pass

def kwta(net,k):
	shunt = 1
	sorted_index = np.flip(np.argsort(net),axis=1)
	winners = sorted_index[0,:k]
	id_k = sorted_index[0,k-1]
	id_kp1 = sorted_index[0,k]
	net_k = net[0,id_k]
	net_kp1 = net[0,id_kp1]
	# kWTA bias term
	q = 0.5
	bias_kWTA = net_kp1 + q * (net_k - net_kp1)
	# net after kwta
	net_after_kWTA = net - bias_kWTA - shunt
	return net_after_kWTA, winners

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class Model():
	def __init__(self,
				x_grid = 16,
				y_grid = 16,
				nA = 4,
				ng=6):
		self.x_grid = x_grid
		self.y_grid = y_grid
		self.ns = x_grid + y_grid
		self.ng = ng
		self.nA = nA
		self.input_size = self.ns
		self.x_vec = np.arange(0,x_grid).reshape(1,-1)
		self.y_vec = np.arange(0,y_grid).reshape(1,-1)
		self.kwta_rate = 0.1
		self.init_network()

	def init_network(self):
		nx = self.input_size
		nh = self.x_grid * self.y_grid
		no = self.nA
		self.nx = nx
		self.nh = nh
		self.no = no
		self.dim_w = {}
		self.dim_w['ih_w'] = (nx,nh)
		self.dim_w['ih_b'] = (1,nh)
		self.dim_w['ho_w'] = (nh,no)

		self.weights = {}
		for g in range(self.ng):
			w = {}
			for key,shape in self.dim_w.items():
				w[key] = 0.1 * ( np.random.random(shape) - 0.5 )
			self.weights[g] = w

	def compute_Q(self, s, g):
		'''compute q(s,g,A;w)'''
		sx = s[0]
		sy = s[1]
		sx_vec = normpdf(self.x_vec, sx, 1)
		sy_vec = normpdf(self.y_vec, sy, 1)
		self.g = g
		self.w = self.weights[g]
		self.x = np.concatenate( (sx_vec, sy_vec), axis=1)
		self.net = self.x @ self.w['ih_w'] + self.w['ih_b']
		k = round(self.kwta_rate * self.nh) 
		self.net_after_kwta, self.winners = kwta(self.net, k)
		self.act = sigmoid(self.net_after_kwta)
		self.Q = self.act @ self.w['ho_w']
		return self.Q

	def compute_Qp(self, sp, g):
		'''compute q(s',g,A;w)'''
		spx = sp[0]
		spy = sp[1]
		spx_vec = normpdf(self.x_vec, spx, 1)
		spy_vec = normpdf(self.y_vec, spy, 1)
		self.g = g
		self.w = self.weights[g]
		self.xp = np.concatenate( (spx_vec, spy_vec), axis=1)
		self.net_prime = self.xp @ self.w['ih_w'] + self.w['ih_b']
		k = round(self.kwta_rate * self.nh) 
		self.net_after_kwta_prime, self.winners_prime = kwta(self.net_prime, k)
		self.act_prime = sigmoid(self.net_after_kwta_prime)
		self.Qp = self.act_prime @ self.w['ho_w']
		return self.Qp

	def update_w(self,a,delta):
		'''update w in q(s,g,A;w)'''
		error = -delta
		self.act_winners = self.act[:,self.winners]		
		delta_j_winners = error * self.w['ho_w'][self.winners,a].reshape(1,-1) \
									 * self.act_winners * (1.0-self.act_winners)
		self.w['ho_w'][self.winners,a] += delta * self.act_winners.reshape(-1,)
		self.w['ih_w'][:,self.winners] += - self.x.T @ delta_j_winners
		self.w['ih_b'][:,self.winners] += - delta_j_winners
		self.weights[self.g] = self.w

	def update_Q_to_Qp(self):
		''' Q <-- Q' '''
		self.Q = copy(self.Qp)
		self.x = copy(self.xp)
		self.winners = copy(self.winners_prime)
		self.act = copy(self.act_prime)

	def save_model(self):
		model_file_path = './models/controller.pkl'
		with open(model_file_path, 'wb') as f:
			pickle.dump(self.weights, f)

	def load_model(self):
		model_file_path = './models/controller.pkl'
		with open(model_file_path, 'rb') as f:
			self.weights = pickle.load(f)
			self.weights = self.weights[0] # comment this later


class MetaModel():
	def __init__(self):
		self.ng = 6
		self.ns = 6
		self.n_rooms = 4
		self.n_outliers = 2
		self.shape = (self.ns, self.ng)
		self.w = 0.1 * ( np.random.random(self.shape) - 0.5 )

	def compute_Q(self,S):
		self.Q = S @ self.w
		self.S = S
		return self.Q

	def compute_QP(self,SP):
		self.QP = SP @ self.w
		return self.QP

	def update_w(self,g_id,delta):
		self.w[:,g_id] += delta * self.S.reshape(-1,)

class VanillaRLModel():
	def __init__(self,
				x_grid = 16,
				y_grid = 16,
				nA = 4):
		self.x_grid = x_grid
		self.y_grid = y_grid
		self.ns = x_grid + y_grid
		self.nA = nA
		self.input_size = self.ns
		self.x_vec = np.arange(0,x_grid).reshape(1,-1)
		self.y_vec = np.arange(0,y_grid).reshape(1,-1)
		self.kwta_rate = 0.1
		self.init_network()

	def init_network(self):
		nx = self.input_size
		nh = self.x_grid * self.y_grid
		no = self.nA
		self.nx = nx
		self.nh = nh
		self.no = no
		self.dim_w = {}
		self.dim_w['ih_w'] = (nx,nh)
		self.dim_w['ih_b'] = (1,nh)
		self.dim_w['ho_w'] = (nh,no)


		self.w = {}
		for key,shape in self.dim_w.items():
			self.w[key] = 0.1 * ( np.random.random(shape) - 0.5 )

	def compute_Q(self, s):
		'''compute q(s,A;w)'''
		sx = s[0]
		sy = s[1]
		sx_vec = normpdf(self.x_vec, sx, 1)
		sy_vec = normpdf(self.y_vec, sy, 1)
		self.x = np.concatenate( (sx_vec, sy_vec), axis=1)
		self.net = self.x @ self.w['ih_w'] + self.w['ih_b']
		k = round(self.kwta_rate * self.nh) 
		self.net_after_kwta, self.winners = kwta(self.net, k)
		self.act = sigmoid(self.net_after_kwta)
		self.Q = self.act @ self.w['ho_w']
		return self.Q

	def compute_Qp(self, sp):
		'''compute q(s',A;w)'''
		spx = sp[0]
		spy = sp[1]
		spx_vec = normpdf(self.x_vec, spx, 1)
		spy_vec = normpdf(self.y_vec, spy, 1)
		self.xp = np.concatenate( (spx_vec, spy_vec), axis=1)
		self.net_prime = self.xp @ self.w['ih_w'] + self.w['ih_b']
		k = round(self.kwta_rate * self.nh) 
		self.net_after_kwta_prime, self.winners_prime = kwta(self.net_prime, k)
		self.act_prime = sigmoid(self.net_after_kwta_prime)
		self.Qp = self.act_prime @ self.w['ho_w']
		return self.Qp

	def update_w(self,a,delta):
		'''update w in q(s,g,A;w)'''
		error = -delta
		self.act_winners = self.act[:,self.winners]		
		delta_j_winners = error * self.w['ho_w'][self.winners,a].reshape(1,-1) \
									 * self.act_winners * (1.0-self.act_winners)
		self.w['ho_w'][self.winners,a] += delta * self.act_winners.reshape(-1,)
		self.w['ih_w'][:,self.winners] += - self.x.T @ delta_j_winners
		self.w['ih_b'][:,self.winners] += - delta_j_winners


# class Model_1():
# 	def __init__(self,
# 				x_grid = 16,
# 				y_grid = 16,
# 				nA = 4):
# 		self.x_grid = x_grid
# 		self.y_grid = y_grid
# 		self.ns = x_grid + y_grid
# 		self.ng = x_grid + y_grid
# 		self.nA = nA
# 		self.input_size = self.ns + self.ng 
# 		self.x_vec = np.arange(0,x_grid).reshape(1,-1)
# 		self.y_vec = np.arange(0,y_grid).reshape(1,-1)
# 		self.kwta_rate = 0.1
# 		self.init_network()

# 	def init_network(self):
# 		nx = self.input_size
# 		nh = 1000
# 		no = self.nA
# 		self.nx = nx
# 		self.nh = nh
# 		self.no = no
# 		self.dim_w = {}
# 		self.dim_w['ih_w'] = (nx,nh)
# 		self.dim_w['ih_b'] = (1,nh)
# 		self.dim_w['ho_w'] = (nh,no)

# 		self.w = {}
# 		for key,shape in self.dim_w.items():
# 			self.w[key] = 0.01 * ( np.random.random(shape) - 0.5 )

# 	def compute_Q(self, s, g):
# 		'''compute q(s,g,A;w)'''
# 		sx = s[0]
# 		sy = s[1]
# 		gx = g[0]
# 		gy = g[1]
# 		sx_vec = normpdf(self.x_vec, sx, 1)
# 		sy_vec = normpdf(self.y_vec, sy, 1)
# 		gx_vec = normpdf(self.x_vec, gx, 1)
# 		gy_vec = normpdf(self.y_vec, gy, 1)

# 		self.x = np.concatenate( (sx_vec, sy_vec, gx_vec, gy_vec), axis=1)

# 		self.net = self.x @ self.w['ih_w'] + self.w['ih_b']
# 		k = round(self.kwta_rate * self.nh) 
# 		self.net_after_kwta, self.winners = kwta(self.net, k)
# 		self.act = sigmoid(self.net_after_kwta)
# 		self.Q = self.act @ self.w['ho_w']
# 		return self.Q

# 	def compute_Qp(self, sp, g):
# 		'''compute q(s',g,A;w)'''
# 		spx = sp[0]
# 		spy = sp[1]
# 		gx = g[0]
# 		gy = g[1]
# 		spx_vec = normpdf(self.x_vec, spx, 1)
# 		spy_vec = normpdf(self.y_vec, spy, 1)
# 		gx_vec = normpdf(self.x_vec, gx, 1)
# 		gy_vec = normpdf(self.y_vec, gy, 1)

# 		self.xp = np.concatenate( (spx_vec, spy_vec, gx_vec, gy_vec), axis=1)

# 		self.net_prime = self.xp @ self.w['ih_w'] + self.w['ih_b']
# 		k = round(self.kwta_rate * self.nh) 
# 		self.net_after_kwta_prime, self.winners_prime = kwta(self.net_prime, k)
# 		self.act_prime = sigmoid(self.net_after_kwta_prime)
# 		self.Qp = self.act_prime @ self.w['ho_w']
# 		return self.Qp

# 	def epsilon_greedy(self, Q, epsilon):
# 		if random() < epsilon:
# 			return randint(0, self.nA-1)
# 		else:
# 			return Q.argmax()

# 	def update_w(self,a,delta):
# 		'''update w in q(s,g,A;w)'''
# 		error = -delta
# 		self.act_winners = self.act[:,self.winners]		
# 		delta_j_winners = error * self.w['ho_w'][self.winners,a].reshape(1,-1) \
# 									 * self.act_winners * (1.0-self.act_winners)
# 		self.w['ho_w'][self.winners,a] += delta * self.act_winners.reshape(-1,)
# 		self.w['ih_w'][:,self.winners] += - self.x.T @ delta_j_winners
# 		self.w['ih_b'][:,self.winners] += - delta_j_winners

# 	def update_Q_to_Qp(self):
# 		''' Q <-- Q' '''
# 		self.Q = copy(self.Qp)
# 		self.x = copy(self.xp)
# 		self.winners = copy(self.winners_prime)
# 		self.act = copy(self.act_prime)


# class QTable():
# 	def __init__(self,
# 				x_grid = 16,
# 				y_grid = 16,
# 				nA = 4):
# 		self.x_grid = x_grid
# 		self.y_grid = y_grid
# 		self.ns = x_grid * y_grid
# 		self.ng = x_grid * y_grid
# 		self.nA = nA
# 		self.init_Q_table()

# 	def init_Q_table(self):
# 		self.Q_table = np.zeros((self.ns,self.ng,self.nA), np.float32)

# 	def convert_to_index(self,s):
# 		return self.x_grid * int(s[0]) + int(s[1])
	
# 	def compute_Q(self,s,g):
# 		id_s = self.convert_to_index(s)
# 		id_g = self.convert_to_index(g)
# 		self.Q = self.Q_table[id_s,id_g,:]
# 		return self.Q
# 	def epsilon_greedy(self, Q, epsilon):
# 		if random() < epsilon:
# 			return randint(0, self.nA-1)
# 		else:
# 			return Q.argmax()
# 	def compute_Qp(self, sp, g):
# 		id_sp = self.convert_to_index(sp)
# 		id_g = self.convert_to_index(g)
# 		self.Qp = self.Q_table[id_sp,id_g,:]
# 		return self.Qp

# 	def update_Q_to_Qp(self):
# 		''' Q <-- Q' '''
# 		self.Q = copy(self.Qp)

# 	def update_Q_table(self,s,g,a,error):
# 		id_s = self.convert_to_index(s)
# 		id_g = self.convert_to_index(g)
# 		self.Q_table[id_s,id_g,a] += error

# class Model_2():
# 	def __init__(self,
# 				x_grid = 16,
# 				y_grid = 16,
# 				nA = 4):
# 		self.x_grid = x_grid
# 		self.y_grid = y_grid
# 		self.ns = x_grid + y_grid
# 		self.ng = x_grid * y_grid
# 		self.nA = nA
# 		self.input_size = self.ns
# 		self.x_vec = np.arange(0,x_grid).reshape(1,-1)
# 		self.y_vec = np.arange(0,y_grid).reshape(1,-1)
# 		self.kwta_rate = 0.1
# 		self.init_network()

# 	def init_network(self):
# 		nx = self.input_size
# 		nh = self.x_grid * self.y_grid
# 		no = self.nA
# 		self.nx = nx
# 		self.nh = nh
# 		self.no = no
# 		self.dim_w = {}
# 		self.dim_w['ih_w'] = (nx,nh)
# 		self.dim_w['ih_b'] = (1,nh)
# 		self.dim_w['ho_w'] = (nh,no)

# 		self.w = {}
# 		for key,shape in self.dim_w.items():
# 			self.w[key] = 0.1 * ( np.random.random(shape) - 0.5 )

# 	def compute_Q(self, s, g):
# 		'''compute q(s,g,A;w)'''
# 		sx = s[0]
# 		sy = s[1]
# 		gx = g[0]
# 		gy = g[1]
# 		sx_vec = normpdf(self.x_vec, sx, 1)
# 		sy_vec = normpdf(self.y_vec, sy, 1)

# 		self.x = np.concatenate( (sx_vec, sy_vec), axis=1)

# 		self.net = self.x @ self.w['ih_w'] + self.w['ih_b']
# 		k = round(self.kwta_rate * self.nh) 
# 		self.net_after_kwta, self.winners = kwta(self.net, k)
# 		self.act = sigmoid(self.net_after_kwta)
# 		self.Q = self.act @ self.w['ho_w']
# 		return self.Q

# 	def compute_Qp(self, sp, g):
# 		'''compute q(s',g,A;w)'''
# 		spx = sp[0]
# 		spy = sp[1]
# 		gx = g[0]
# 		gy = g[1]
# 		spx_vec = normpdf(self.x_vec, spx, 1)
# 		spy_vec = normpdf(self.y_vec, spy, 1)
# 		gx_vec = normpdf(self.x_vec, gx, 1)
# 		gy_vec = normpdf(self.y_vec, gy, 1)

# 		self.xp = np.concatenate( (spx_vec, spy_vec, gx_vec, gy_vec), axis=1)

# 		self.net_prime = self.xp @ self.w['ih_w'] + self.w['ih_b']
# 		k = round(self.kwta_rate * self.nh) 
# 		self.net_after_kwta_prime, self.winners_prime = kwta(self.net_prime, k)
# 		self.act_prime = sigmoid(self.net_after_kwta_prime)
# 		self.Qp = self.act_prime @ self.w['ho_w']
# 		return self.Qp

# 	def epsilon_greedy(self, Q, epsilon):
# 		if random() < epsilon:
# 			return randint(0, self.nA-1)
# 		else:
# 			return Q.argmax()

# 	def update_w(self,a,delta):
# 		'''update w in q(s,g,A;w)'''
# 		error = -delta
# 		self.act_winners = self.act[:,self.winners]		
# 		delta_j_winners = error * self.w['ho_w'][self.winners,a].reshape(1,-1) \
# 									 * self.act_winners * (1.0-self.act_winners)
# 		self.w['ho_w'][self.winners,a] += delta * self.act_winners.reshape(-1,)
# 		self.w['ih_w'][:,self.winners] += - self.x.T @ delta_j_winners
# 		self.w['ih_b'][:,self.winners] += - delta_j_winners

# 	def update_Q_to_Qp(self):
# 		''' Q <-- Q' '''
# 		self.Q = copy(self.Qp)
# 		self.x = copy(self.xp)
# 		self.winners = copy(self.winners_prime)
# 		self.act = copy(self.act_prime)


