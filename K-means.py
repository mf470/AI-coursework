from functools import reduce


def setCentroids(clusters):
	"""
	Sets new centroids for the clusters
	:param clusters: List of list of elements
	:return: list of centroids
	"""
	centroids = []
	for cluster in clusters:
		centroids += [reduce(lambda x, y: x + y, cluster) / len(cluster)]

	return centroids


def setClusters(clusters, centroids):
	"""
	Re-sorts the elements to be in the cluster with the centroid closest to it
	:param clusters: list of list of elements
	:param centroids: the list of current centroids
	:return: null
	"""
	for cluster in range(len(clusters)):
		for element in clusters[cluster]:
			min = abs(element - centroids[0])
			index = 0
			for i in range(1, len(centroids)):
				if abs(element - centroids[i]) < min:
					min = abs(element - centroids[i])
					index = i
			if index != cluster:
				clusters[cluster].remove(element)
				clusters[index] += [element]


if __name__ == "__main__":
	clusters = [[1, 2, 3], [11, 4, 7], [10, 22]]
	for i in range(4):
		centroids = setCentroids(clusters)
		print('-' * 80)
		print('centroids = ', centroids)
		print('clusters = ', clusters)
		setClusters(clusters, centroids)
