from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
import os
import pickle 

class SubgoalDiscovery():
	def __init__(self,n_clusters=5,**kargs):
		self.n_clusters = n_clusters
		self.outliers = []
		self.centroid_memory= [] # all the centroids over time of learning
		self.centroid_subgoals = [] # list of recent centroids
		self.G = [] # recent list of all subgoals
		self.C = None # Kmeans centroids in numpy arrays rounded

	def feed_data(self,X):
		self.X = X

	def find_kmeans_clusters(self):
		if self.C is None:
			print('first time of using kmeans to find centroids')
			init = 'random' 
		else:
			print('updating Kmeans centroids using previous centroids')
			init = self.C
		self.kmeans = KMeans(n_clusters=self.n_clusters,init=init,max_iter=300)
		self.kmeans.fit(self.X)
		self.C = self.cluster_centroids()
		self.centroid_memory.append(self.C)		
		self.centroid_subgoals = [ tuple(g) for g in list(self.C) ]
		self.G = self.centroid_subgoals + self.outliers

	def cluster_centroids(self):
		return np.round_(self.kmeans.cluster_centers_).astype(int)

	def push_outlier(self, outlier,threshold=16):
		if len(self.outliers) == 0:
			self.outliers.append(outlier)
			self.G = self.centroid_subgoals + self.outliers
			print('discovered outlier is added to the outliers list')
			return
		distance = []
		for member in self.outliers:
			distance.append( (member[0]-outlier[0])**2+(member[1]-outlier[1])**2 )
		if min(distance) >= threshold:
			self.outliers.append(outlier)
			self.G = self.centroid_subgoals + self.outliers
		else:
			print('discovered outlier already in the outliers list')

	def save_results(self,results_file_path='./results/subgoals.pkl'):
		if not os.path.exists("results"):
			os.makedirs("results")
		with open(results_file_path, 'wb') as f: 
			pickle.dump([self.centroid_subgoals,self.outliers], f)

class UnsupervisedOutlierDetection():
	def __init__(self,kernel='rbf'):
		self.clf = svm.OneClassSVM(nu=0.1, kernel=kernel, gamma=0.1)

	def fit_data(self,X):
		self.clf.fit(X)

	def detect_outlier(self,x):
		if self.clf.predict(x) < 0:
			return True
		else:
			return False 



