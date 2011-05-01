from numpy import *

import matplotlib.pyplot as plt

from pprint import pprint

class KMeans:
	"""
	Fairly faithful implementation of K-Means as described in the book
	Introduction to Information Retrieval, chapter 16 Flat Clustering,
	subsection K-Means:
	
	http://nlp.stanford.edu/IR-book/html/htmledition/k-means-1.html
	
	Numpy is used to make computations easier.
	"""

	def __init__(self, K, N, R=2):
		self.K = K
		self.points = random.rand(N, R)
		self.centroids = random.rand(K, R)
	
	def _assign_points(self):
		# Default clusters are default.
		clusters = [[] for i in range(self.K)]
		
		for point in self.points:
			# Figure out which centroid has the shortest distance to the point.
			pos = argmin([linalg.norm(point-centroid) for centroid in self.centroids])
			clusters[pos].append(point)
			
		return clusters
	
	def cluster(self, cb=None):
		clusters = self._assign_points()
		
		# Perform 5 iterations.
		for a in range(10):
			# Recompute centroid.
			for i_c in range(len(clusters)):
				self.centroids[i_c][0] = mean(clusters[i_c][0::2])
				self.centroids[i_c][1] = mean(clusters[i_c][1::2])
			
			# Recompute clustering.
			clusters = self._assign_points()
			
			if cb:
				cb(self.centroids)
			
		return clusters

class PlotKMeans:
	def __init__(self, k):
		self.k = k
		
		plt.ion()
		plt.xlim(0, 1)
		plt.ylim(0, 1)
		
		self.colours = ["b", "g", "r", "c", "m", "y"]
	
	def plot(self):
		c = k.cluster(cb=self._plot_centroids)
		self._plot_clusters(c)
	
	def _plot_centroids(self, centroids):
		for (i, centroid) in enumerate(centroids):
			plt.plot(centroid[0::2],
				centroid[1::2], "+",
				color=self.colours[i % len(self.colours)])
			plt.draw()
	
	def _plot_clusters(self, clusters):
		for cluster in enumerate(clusters):
			for point in cluster[1]:
				plt.plot(point[0::2],
					point[1::2], ".", alpha=0.5,
					color=self.colours[cluster[0] % len(self.colours)])
		
		plt.draw()

k = KMeans(3, 500)
p = PlotKMeans(k)
p.plot()

input()