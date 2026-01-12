# This program outputs a visualization of a K-Mean cluster.
# Refer to comments for specific steps

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# FUNCTIONS
# Returns the distance between two points
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Assigns each point to the nearest cluster center
def assign_clusters(X, clusters):
    for inx in range(X.shape[0]):
        dist = []

        curr_x = X[inx]

        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)                                # Appends distance to list
        curr_cluster = np.argmin(dist)                      # Finds index of minimum distance
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

# Updates cluster centers based on assigned points and means
def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = np.mean(points, axis=0)            # Calculates mean of points
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters
    
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))                        # Assigns the index of the closest cluster
    return pred
# END FUNCTIONS

# DATASET
# Form blobs in the dataset
X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)

# Initialize clusters
k = 3

clusters = {}
np.random.seed(23)

for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }
    
    clusters[idx] = cluster

clusters = assign_clusters(X,clusters)
clusters = update_clusters(X,clusters)
pred = pred_cluster(X,clusters)

# Plot initial clusters and centers
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1],c=pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker='^',c='red')
plt.show()

