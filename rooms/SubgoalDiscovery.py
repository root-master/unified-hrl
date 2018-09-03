from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn import svm

class SubgoalDiscovery():
	def __init__(self,n_clusters=8,**kargs):
		self.n_clusters = n_clusters
		
	def feed_data(self,X):
		self.X = X

	def find_kmeans_clusters(self,init='kmean++'):
		self.kmeans = KMeans(n_clusters=self.n_clusters,init=init,max_iter=300)
		self.kmeans.fit(self.X)

	def find_gaussian_clusters(self):
		self.gaussian = GaussianMixture(n_components=4).fit(self.X)

	def find_kmeans_clusters_online(self,init='k-means++'):
		self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,init=init,max_iter=300).fit(self.X)

	def cluster_centroids(self):
		return np.round_(self.kmeans.cluster_centers_*16+0.5)

	def predict_closest_cluster(self, s):
		c_id = self.predict_closest_cluster_index(s)
		centroids = self.kmeans.cluster_centers_
		c = centroids[:,c_id]
		return np.round_(c*16+0.5)

	def predict_closest_cluster_index(self, s):
		z = np.array(list(s)).reshape(1,-1)/16
		return self.kmeans.predict(z)

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



