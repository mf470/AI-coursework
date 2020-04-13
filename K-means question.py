from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random as rnd
import numpy as np


def matrixNorm(A, B):
	"""
	A norm or 'distance' function for matrices
	:param A: n by n matrix
	:param B: n by n matrix
	:return: The norm of the difference of the matrices
	"""
	norm = 0
	diff = [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
	for i in range(len(A)):
		for j in range(len(A[0])):
			norm += diff[i][j]*diff[i][j]
	return norm


def mean(matrices):
	"""

	:param matrices:
	:return:
	"""
	mean = [[0]*len(matrices[0][0])]*len(matrices[0])
	for i in range(len(matrices[0])):
		for j in range(len(matrices[0][0])):
			mean[i][j] = sum(matrix[i][j] for matrix in matrices)/len(matrices)
	return mean


def setCentroids(clusters):
	"""

	:param clusters:
	:return:
	"""
	centroids = []
	for cluster in clusters:
		centroids.append(np.array(mean(cluster)))
	return centroids


def setClusters(dataset, centroids):
	"""

	:param dataset:
	:param centroids:
	:return:
	"""
	clusters = [[] for i in range(len(centroids))]
	for matrix in dataset:
		min = matrixNorm(centroids[0], matrix)
		index = 0
		for i in range(1, len(centroids)):
			diff = matrixNorm(centroids[i], matrix)
			if diff < min:
				min, index = diff, i
		(clusters[index]).append(matrix)

	return clusters


def k_means(dataset, k):
	"""

	:param dataset:
	:param k:
	:return:
	"""
	i = 0
	centroids = rnd.sample(dataset, k)
	prevCentroids = []
	while not np.array_equal(centroids, prevCentroids):
		i += 1
		prevCentroids = centroids
		clusters = setClusters(dataset, centroids)
		centroids = setCentroids(clusters)
	return clusters


if __name__ == "__main__":
	digits = load_digits()
	trainingSet = [digits.data[i] for i in range(len(digits.images)) if i % 6 != 0]
	testSet = [digits.data[i] for i in range(len(digits.images)) if i % 6 == 0]
	#centroids, clusters = k_means(trainingSet, 10)
	#testAssignment = setClusters(testSet, centroids)
	# plt.gray()
	# for i in range(10):
	# 	plt.matshow(testAssignment[i][0])
	# 	plt.show()
	# 	plt.matshow(clusters[i][0])
	# 	plt.show()

	wcss = []
	for i in range(1, 20):
		kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=0)
		kmeans.fit(trainingSet)
		wcss.append(kmeans.inertia_)
	plt.plot(range(1, 20), wcss)
	plt.show()
