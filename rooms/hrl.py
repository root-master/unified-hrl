from Model import Model, MetaModel
import numpy as np
class Controller():
	def __init__(self):
		self.Q = Model()

class MetaController():
	def __init__(self,				
				subgoal_discovery=None,
				outliers=None):
		self.ng = 6
		self.ns = 6
		self.subgoal_discovery = subgoal_discovery
		self.outliers = outliers
		self.n_rooms = 4
		self.Q = MetaModel()

	def get_meta_state(self,s,has_key=False):
		S = np.zeros((1,self.ng))
		if s in self.outliers and has_key:
			S_id = self.outliers.index(s)
			S_id = S_id + self.n_rooms
		else:
			S_id = self.subgoal_discovery.predict_closest_cluster_index(s)
		S[0,S_id] = 1
		return S
