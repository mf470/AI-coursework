from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def computeNorm(dataset, centroids):
	"""
	Computes the distance from each element in the dataset to each centroid
	:param dataset: (ndarray) Dataset being clustered
	:param centroids: (ndarray) List of centroids
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
	:param dataset: (ndarray) Dataset being clustered
	:param centroids: (ndarray) List of centroids
	:return: (ndarray) List of labels of which centroid each element is closest to
	"""
	norm = computeNorm(dataset, centroids)
	return np.argmin(norm, axis=1)


def setCetroids(dataset, labels, k):
	"""
	Calculates the centroids for each cluster
	:param dataset: (ndarray) Dataset being clustered
	:param labels: (ndarray) List assigning each element to a cluster
	:param k: (int) The number of clusters
	:return: (ndarray) Centroids
	"""
	centroids = np.zeros((k, dataset.shape[1]))
	for i in range(k):
		centroids[i, :] = np.mean(dataset[labels == i, :], axis=0)
	return centroids


def calcError(dataset, labels, centroids):
	"""
	Calculates the Sum of squared error
	:param dataset: (ndarray) Dataset being clustered
	:param labels: (ndarray) List assigning each element to a cluster
	:param centroids: ndarray of centroids
	:return: (float) error
	"""
	distance = np.zeros(dataset.shape[0])
	for i in range(dataset.shape[0]):
		distance[i] = norm(dataset[i] - centroids[labels[i]])
	return np.sum(np.square(distance))


def k_means(dataset, k, maxit=300):
	"""
	Clusters the dataset into k clusters
	:param dataset: (ndarray) Dataset being clustered
	:param k: (int) The number of clusters required
	:param maxit: (int) Maximum number of iterations to cluster
	:return: (ndarray, ndarray, float) labels of which cluster each element in the dataset belongs to, the centroids of
		the dataset, the sum of squared error
	"""
	centroids = dataset[np.random.permutation(dataset.shape[0])[:k]]
	for i in range(maxit):
		prevCentroids = centroids
		labels = setClusters(dataset, prevCentroids)
		centroids = setCetroids(dataset, labels, k)
		error = calcError(dataset, labels, centroids)
		if np.all(centroids == prevCentroids):
			print(i)
			return labels, centroids, error
	return labels, centroids, error


if __name__ == "__main__":
	digits = load_digits()
	trainingSet = [digits.data[i] for i in range(len(digits.images)) if i % 6 != 0]
	testSet = [digits.data[i] for i in range(len(digits.images)) if i % 6 == 0]
	plt.gray()
	labels, centroids, error = k_means(digits.data, 10)
	print(type(error))
	for i in range(10):
		plt.matshow(centroids[i].reshape((8, 8)))
		plt.show()

	# sse = []
	# for i in range(1, 16):
	# 	labels, centroids, error = k_means(digits.data, i)
	# 	sse.append(error)
	# plt.plot(range(1, 16), sse)
	# plt.show()
