from sklearn.cluster import KMeans
import csv
import numpy as np
import matplotlib.pyplot as plt

dataset = []
with open('pokemon.csv', 'r', encoding='utf-8') as file:
	reader = csv.reader(file)

	next(reader)

	for line in reader:
		# dataset.append(line[1:24] + line[25:27] + [line[28]] + line[33:36])
		dataset.append(line[1:19])
# Remove Minior from dataset
del dataset[773]


# Normalise each column
dataset = np.array(dataset, dtype=float)
maxVals = [max(dataset[:, i]) for i in range(dataset.shape[1])]
for pokemon in dataset:
	for i in range(dataset.shape[1]):
		pokemon[i] /= maxVals[i]

# kmeans = KMeans(n_clusters=18)
# kmeans.fit(dataset)

# with open('cluster.csv', 'w') as file:
# 	writer = csv.writer(file, delimiter=',')
#
# 	for cluster in kmeans.labels_.tolist():
# 		writer.writerow([cluster])


sse = []
for i in range(5, 25):
	kmeans = KMeans(n_clusters=i).fit(dataset)
	sse.append(kmeans.inertia_)
plt.plot(range(5, 25), sse)
plt.xticks(range(5, 25))
plt.grid(True)
plt.xlabel("Clusters")
plt.ylabel("Sum of Squared Error")
plt.show()