from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np

class SubgoalDiscovery():
	def __init__(self,n_clusters=10,**kargs):
		self.n_clusters = n_clusters
		
	def feed_data(self,X):
		self.X = X

	def find_kmeans_clusters(self,init='random'):
		self.kmeans = KMeans(n_clusters=self.n_clusters,init=init,max_iter=300)
		self.kmeans.fit(self.X)

	def cluster_centroids(self):
		return np.round_(self.kmeans.cluster_centers_)

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



