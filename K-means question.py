from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def computeNorm(dataset, centroids):
	distance = np.zeros((dataset.shape[0], len(centroids)))
	for i in range(len(centroids)):
		rowNorm = norm(dataset-centroids[i, :], axis=1)
		distance[:, i] = np.square(rowNorm)
	return distance


def setClusters(dataset, centroids):
	norm = computeNorm(dataset, centroids)
	return np.argmin(norm, axis=1)


def setCetroids(dataset, labels, k):
	centroids = np.zeros((k, dataset.shape[1]))
	for i in range(k):
		centroids[i, :] = np.mean(dataset[labels == i, :], axis=0)
	return centroids


def calcError(dataset, labels, centroids):
	distance = np.zeros(dataset.shape[0])
	for i in range(dataset.shape[0]):
		distance[i] = norm(dataset[i] - centroids[labels[i]])
	return np.sum(np.square(distance))


def k_means(dataset, k, maxit=300):
	centroids = dataset[np.random.permutation(dataset.shape[0])[:k]]
	for i in range(maxit):
		prevCentroids = centroids
		labels = setClusters(dataset, prevCentroids)
		centroids = setCetroids(dataset, labels, k)
		if np.all(centroids == prevCentroids):
			return labels, centroids, error
		error = calcError(dataset, labels, centroids)


if __name__ == "__main__":
	digits = load_digits()
	trainingSet = [digits.data[i] for i in range(len(digits.images)) if i % 6 != 0]
	testSet = [digits.data[i] for i in range(len(digits.images)) if i % 6 == 0]
	# labels, centroids, error = k_means(digits.data, 10)
	# plt.gray()
	# for i in range(10):
	# 	plt.matshow(centroids[i].reshape((8, 8)))
	# 	plt.show()

	wcss = []
	for i in range(1, 16):
		labels, centroids, error = k_means(digits.data, i)
		wcss.append(error)
	plt.plot(range(1, 16), wcss)
	plt.show()
