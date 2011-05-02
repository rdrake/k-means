from numpy import *

from matplotlib.pyplot import *

from pprint import pprint

class KMeans:
	"""
	Fairly faithful implementation of K-Means as described in the book
	Introduction to Information Retrieval, chapter 16 Flat Clustering,
	subsection K-Means:
	
	http://nlp.stanford.edu/IR-book/html/htmledition/k-means-1.html
	
	Numpy is used to make computations easier.
	"""

	def __init__(self, K, N):
		self.K = K
		self.points = random.randn(N, 2)
		self.centroids = random.randn(K, 2)
	
	def _assign_points(self):
		# Default clusters are default.
		clusters = [[] for i in range(self.K)]
		
		for point in self.points:
			# Figure out which centroid has the shortest distance to the point.
			pos = argmin([linalg.norm(point-centroid) for centroid in self.centroids])
			clusters[pos].append(point)
			
		return clusters
	
	def _compute_rss(self, centroid, cluster):
		return sum([linalg.norm(centroid-cluster) ** 2 for point in cluster])
	
	def cluster(self, cb=None):#cen_cb=None, cst_cb=None):
		clusters = self._assign_points()
		prev_rss = 0
		iters = 0
		
		# Perform 5 iterations.
		while True:
			iters += 1
			rss = sum([self._compute_rss(centroid, clusters[i]) for (i, centroid) in enumerate(self.centroids)])
			
			print "Computed RSS is %.04f." % rss
			
			if rss == prev_rss:
				print "Converged in %d iterations." % iters
				break
			else:
				prev_rss = rss
			
			# Recompute centroid.
			for i_c in range(len(clusters)):
				self.centroids[i_c][0] = mean([p[0] for p in clusters[i_c]])
				self.centroids[i_c][1] = mean([p[1] for p in clusters[i_c]])
			
			# Recompute clustering.
			clusters = self._assign_points()
			
			if cb:
				cb(clusters, self.centroids)
			
		return clusters

class PlotKMeans:
	def __init__(self):
		ion()
		xlim(-3, 3)
		ylim(-3, 3)
		
		self.colours = ["b", "g", "r", "c", "m", "y"]
	
	def plot(self, clusters, centroids):
		hold(False)
		plot([p[0] for p in centroids], [p[1] for p in centroids],
			"+", color="k")
		
		hold(True)
		for (i, cluster) in enumerate(clusters):
			plot([p[0] for p in cluster], [p[1] for p in cluster],
				".", alpha=0.4, color=self.colours[i % len(self.colours)])
		
		draw()

k = KMeans(3, 500)
p = PlotKMeans()

k.cluster(cb=p.plot)

print "Press any key to exit..."
raw_input()