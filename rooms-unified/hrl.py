from Model import Model, MetaModel
import numpy as np

class Controller():
	def __init__(self,subgoal_discovery=None):
		self.subgoal_discovery = subgoal_discovery
		ng = len(subgoal_discovery.G)
		self.Q = Model(ng=ng)

class MetaController():
	def __init__(self,				
				subgoal_discovery=None):

		self.subgoal_discovery = subgoal_discovery
		self.ng = len(subgoal_discovery.G)
		self.ns = self.ng

		self.outliers = subgoal_discovery.outliers
		self.n_clusters = subgoal_discovery.n_clusters
		self.Q = MetaModel(ng=self.ng)

	def get_meta_state(self,s,has_key=False):
		S = np.zeros((1,self.ng))
		if s in self.outliers and has_key:
			S_id = self.outliers.index(s)
			S_id = S_id + self.n_clusters
		else:
			S_id = self.subgoal_discovery.predict_closest_cluster_index(s)
		S[0,S_id] = 1
		return S
