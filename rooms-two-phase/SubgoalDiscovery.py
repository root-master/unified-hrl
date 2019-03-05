from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn import svm
import numpy as np

class SubgoalDiscovery():
	def __init__(self,
				n_clusters=4,
				experience_memory=None,
				kmeans=None,
				**kwargs):
		self.n_clusters = n_clusters
		self.experience_memory = experience_memory
		self.outliers = []
		self.centroid_memory= [] # all the centroids over time of learning
		self.centroid_subgoals = [] # list of recent centroids
		self.G = [] # recent list of all subgoals
		self.C = None # Kmeans centroids in numpy arrays rounded
		self.X = None
		self.kmeans = kmeans
		self.doorways = []
		self.doorway_pairs = []

		self.__dict__.update(kwargs)
		
	def feed_data(self,X):
		self.X = X

	def find_kmeans_clusters(self):
		if self.X is None and self.experience_memory is None:
			print('Error! No data to work with, either feed_data or pass memory')

		if self.experience_memory is not None:
			self.X = self.experience_memory.X

		if self.C is None:
			print('first time of using kmeans to find centroids')
			init = 'random' 
		else:
			print('updating Kmeans centroids using previous centroids')
			init = self.C
		if self.kmeans is None:
			self.kmeans = KMeans(n_clusters=self.n_clusters,init=init,max_iter=300)
		self.kmeans.fit(self.X)
		self.C = self.cluster_centroids()
		self.centroid_memory.append(self.C)		
		self.centroid_subgoals = [ tuple(g) for g in list(self.C) ]
		self.G = self.centroid_subgoals + self.outliers

	def find_kmeans_clusters_random_seed(self):
		self.X = self.experience_memory.X
		self.kmeans = KMeans(n_clusters=self.n_clusters,init='random',max_iter=300)
		self.kmeans.fit(self.X)
		self.C = self.cluster_centroids()
		self.centroid_memory.append(self.C)		
		self.centroid_subgoals = [ tuple(g) for g in list(self.C) ]
		self.G = self.centroid_subgoals + self.outliers

	def find_gaussian_clusters(self):
		self.gaussian = GaussianMixture(n_components=4).fit(self.X)

	def find_kmeans_clusters_online(self,init='k-means++'):
		self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,init=init,max_iter=300).fit(self.X)

	def cluster_centroids(self):
		return np.round_(self.kmeans.cluster_centers_*16+0.5).astype(int)

	def predict_closest_cluster(self, s):
		c_id = self.predict_closest_cluster_index(s)
		centroids = self.kmeans.cluster_centers_
		c = centroids[:,c_id]
		return np.round_(c*16+0.5)

	def predict_closest_cluster_index(self, s):
		z = np.array(list(s)).reshape(1,-1)/16
		return self.kmeans.predict(z)

	def push_outlier(self, outlier,threshold=1):
		if len(self.outliers) == 0:
			self.outliers.append(outlier)
			self.G = self.centroid_subgoals + self.outliers
			# print('discovered outlier is added to the outliers list')
			return
		distance = []
		for member in self.outliers:
			distance.append( (member[0]-outlier[0])**2+(member[1]-outlier[1])**2 )
		if min(distance) >= threshold:
			self.outliers.append(outlier)
			self.G = self.centroid_subgoals + self.outliers
			print('outlier discovered: ', outlier)
		else:
			print('discovered outlier already in the outliers list')

	def push_doorways(self,e,threshold=5):
		s = e[0]
		a = e[1]
		r = e[2]
		sp =e[3]
		room_1 = self.predict_closest_cluster_index(s)
		room_2 = self.predict_closest_cluster_index(sp)
		if room_1 != room_2:
			if len(self.doorways)==0:
				self.doorways.append(s)
				self.doorways.append(sp)
				self.doorway_pairs.append([s,sp])
				print('doorways discovered: ', s, 'and',sp)
			else:
				distance = []
				for member in self.doorways:
					distance.append( (member[0]-s[0])**2+(member[1]-s[1])**2 )
				if min(distance) >= threshold:
					self.doorways.append(s)
					self.doorways.append(sp)
					print('doorways discovered: ', s, 'and',sp)
					self.doorway_pairs.append([s,sp])
				# else:
				# 	print('discovered doorways already in the doorways list')

	def report(self):
		print('outliers: ', self.outliers) 
		print('centroids: ', self.centroid_subgoals)
		print('doorways: ', self.doorways)

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



