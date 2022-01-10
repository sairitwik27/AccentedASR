from nltk.cluster import KMeansClusterer
import nltk

# make the matrix with the words
words = acc_dict.keys()
X = []
for w in words:
    X.append(acc_dict[w].numpy())


# perform the clustering on the matrix
NUM_CLUSTERS=20
kclusterer = KMeansClusterer(NUM_CLUSTERS,distance=nltk.cluster.util.cosine_distance)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

labels = {}
keys = list(words)
# print the cluster each word belongs
for i in range(len(X)):
    labels[keys[i]] = assigned_clusters[i]

import json

with open('accent_labels.json', 'w') as fp:
    json.dump(labels, fp, sort_keys=True, indent=4)