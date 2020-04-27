from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import (silhouette_score, silhouette_samples)


def computeNorm(dataset, centroids):
	"""
	Computes the distance from each element in the dataset to each centroid
	:param dataset: array[n_samples, n_features] Dataset being clustered
	:param centroids: array[k, n_features] List of centroids
	:return: (ndarray) List of distance of each element in the dataset to each centroid
	"""
	distance = np.zeros((dataset.shape[0], len(centroids)))
	for i in range(len(centroids)):
		rowNorm = norm(dataset-centroids[i, :], axis=1)
		distance[:, i] = np.square(rowNorm)
	return distance


def setClusters(dataset, centroids):
	"""
	Finds which centroid each element in the dataset is closest to
	:param dataset: array[n_samples, n_features] Dataset being clustered
	:param centroids: array[k, n_features] List of centroids
	:return: array[k, n_features] List of labels of which centroid each element is closest to
	"""
	norm = computeNorm(dataset, centroids)
	return np.argmin(norm, axis=1)


def setCetroids(dataset, labels, k):
	"""
	Calculates the centroids for each cluster
	:param dataset: array[n_samples, n_features] Dataset being clustered
	:param labels: array[n_samples] List assigning each element to a cluster
	:param k: (int) The number of clusters
	:return: array[k, n_features] Centroids
	"""
	centroids = np.zeros((k, dataset.shape[1]))
	for i in range(k):
		centroids[i, :] = np.mean(dataset[labels == i, :], axis=0)
	return centroids


def calcError(dataset, labels, centroids):
	"""
	Calculates the Sum of squared error
	:param dataset: array[n_samples, n_features] Dataset being clustered
	:param labels: array[n_samples] List assigning each element to a cluster
	:param centroids: array[k, n_features] of centroids
	:return: (float) error
	"""
	distance = np.zeros(dataset.shape[0])
	for i in range(dataset.shape[0]):
		distance[i] = norm(dataset[i] - centroids[labels[i]])
	return np.sum(np.square(distance))


def k_means(dataset, k, maxit=300):
	"""
	Clusters the dataset into k clusters
	:param dataset: array[n_samples, n_features] Dataset being clustered
	:param k: (int) The number of clusters required
	:param maxit: (int) Maximum number of iterations to cluster
	:return: (array[n_samples], array[k, n_features], float) labels of which cluster each element in the dataset belongs
	to, the centroids of the dataset, the sum of squared error
	"""
	centroids = dataset[np.random.permutation(dataset.shape[0])[:k]]
	for i in range(maxit):
		prevCentroids = centroids
		labels = setClusters(dataset, prevCentroids)
		centroids = setCetroids(dataset, labels, k)
		error = calcError(dataset, labels, centroids)
		if np.all(centroids == prevCentroids):
			# print(i)
			return labels, centroids, error
	return labels, centroids, error


def makeSilhouettePlot(dataset, labels, k):
	"""
	Displays the silhouette plot for the clustering of dataset wth labels
	Adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
	:param dataset: array[n_samples, n_features] data that is clustered
	:param labels: array[n_samples] labels of cluster for each sample in the dataset
	:param k: (int) number of clusters
	:return: null
	"""
	fig, ax = plt.subplots()
	average = silhouette_score(dataset, labels)
	sampleValues = silhouette_samples(dataset, labels)
	lower = 10
	for i in range(k):
		clusterSilhouetteValues = sampleValues[labels == i]
		clusterSilhouetteValues.sort()

		clusterSize = clusterSilhouetteValues.shape[0]
		upper = lower + clusterSize
		colour = cm.nipy_spectral(float(i) / k)

		ax.fill_betweenx(np.arange(lower, upper), 0, clusterSilhouetteValues, facecolor=colour, edgecolor=colour, alpha=0.7)

		ax.text(-0.05, lower + 0.5 * clusterSize, str(i))

		lower = upper + 10

	ax.set_title("Silhouette analysis for digit recognition with %d clusters" % k)
	ax.set_xlabel("The silhouette coefficient values")
	ax.set_ylabel("Cluster label")

	# The vertical line for average silhouette score of all the values
	ax.axvline(x=average, color="red", linestyle="--")

	ax.set_yticks([])
	ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

	plt.show()


if __name__ == "__main__":
	digits = load_digits()

	'''Find the elbow plot and silhouette plots'''
	# sse = []
	# for i in range(1, 16):
	# 	labels, centroids, error = k_means(digits.data, i)
	# 	sse.append(error)
	# 	if i == 1:
	# 		continue
	# 	makeSilhouettePlot(digits.data, labels, i)
	# plt.plot(range(1, 16), sse)
	# plt.xticks(range(1, 16))
	# plt.grid(True)
	# plt.xlabel("Clusters")
	# plt.ylabel("Sum of Squared Error")
	# plt.show()

	'''Do clustering with 10 clusters'''
	plt.gray()
	labels, centroids, error = k_means(digits.data, 10)

	accuracy = []
	for i in range(10):
		plt.matshow(centroids[i].reshape((8, 8)))
		plt.show()
		actual = digits.target[labels == i]
		unique, counts = np.unique(actual, return_counts=True)
		accuracy.append(max(counts)/sum(counts))

	print(sum(accuracy)/10)

	'''Find average accuracy of clustering'''
	# accuracies = []
	# for j in range(1000):
	# 	labels, centroids, error = k_means(digits.data, 10)
	# 	accuracy = []
	# 	for i in range(10):
	# 		actual = digits.target[labels == i]
	# 		unique, counts = np.unique(actual, return_counts=True)
	# 		accuracy.append(max(counts)/sum(counts))
	#
	# 	accuracies.append(sum(accuracy)/10)
	# print(sum(accuracies)/len(accuracies))
