from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

class SubgoalDiscovery():
	def __init__(self,n_clusters=8,**kargs):
		self.n_clusters = n_clusters

	def feed_data(self,X):
		self.X = X

	def find_kmeans_clusters(self,init='k-means++'):
		self.kmeans = KMeans(n_clusters=self.n_clusters,init=init,max_iter=300).fit(self.X)

	def find_gaussian_clusters(self):
		self.gaussian = GaussianMixture(n_components=4).fit(self.X)


	def find_kmeans_clusters_online(self,init='k-means++'):
		self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,init=init,max_iter=300).fit(self.X)




