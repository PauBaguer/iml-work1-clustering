import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


#####################################
#         Metric definitions        #
#####################################


# Compute the Manhattan distance between each point in point_vector and the variable point.
def l1_distance(point, point_vector):
    return np.sqrt(np.sum(np.absolute(point - point_vector), axis=1))


# Compute the Euclidean distance between each point in point_vector and the variable point.
def l2_distance(point, point_vector):
    return np.sqrt(np.sum((point - point_vector) ** 2, axis=1))


# Compute the cosine similarity distance between each point in point_vector and the variable point.
def cosine_distance(point, point_vector):
    dot_product = np.dot(point_vector, point)
    norm_x = np.sqrt(np.sum(point ** 2))
    norm_y = np.sqrt(np.sum(point_vector ** 2, axis=1))
    return 1 - dot_product / (norm_x * norm_y)


metrics = {
    'l1': l1_distance,
    'l2': l2_distance,
    'cosine': cosine_distance,
}


#####################################
#           Initializations         #
#####################################

# Random initialization
def random_init(data, num_clusters, metric):
    return data[np.random.choice(data.shape[0], num_clusters, replace=False)]


# Use kmeans++ for the initialization
def kmeanspp_init(data, num_clusters, metric):
    # Choose one center uniformly at random
    centroids = [random.choice(data)]
    for _ in range(num_clusters - 1):
        # Compute the distances from points to the centroids
        prob = np.min([metric(c, data) ** 2 for c in centroids], axis=0)
        # Normalize the vector of probabilities
        prob /= np.sum(prob)
        # Choose another centroid based on the probabilities stored in prob
        index, = np.random.choice(range(len(data)), size=1, p=prob)
        centroids += [data[index]]
    return centroids


initializations = {
    'random': random_init,
    'kmeans++': kmeanspp_init,
}

#####################################
#              Clustering           #
#####################################


def kmeans(data, num_clusters, max_iter=5000, metric='l2', centroid_init='kmeans++'):
    # Choose the metric to be used
    metric_function = metrics[metric]
    init_function = initializations[centroid_init]

    # Choose the initial centroids
    centroids = init_function(data, num_clusters, metric_function)

    i = 0
    prev_centroids = None
    # Iterate until the clusters stabilize, or we reach the maximum of iterations
    while np.not_equal(centroids, prev_centroids).any() and i < max_iter:
        clusters = [[] for _ in range(num_clusters)]
        # Assign each point to the cluster with the closest centroid
        for row in data:
            distance = [metric_function(row, centroids)]
            index = np.argmin(distance)
            clusters[index].append(row)

        # Assign the used centroids to the previous ones
        prev_centroids = centroids
        # Recompute the new centroids
        centroids = [np.mean(cpoints, axis=0) for cpoints in clusters]
        # Handle empty clusters
        # It could be improved by choosing the centroid from the cluster with highest SSE
        # or choosing the point that is farthest away from any current centroid
        for i in range(num_clusters):
            if np.isnan(centroids[i]).any():
                centroids[i] = prev_centroids[i]
        i += 1
    # Compute the final cluster assignation for the data
    assigned_cluster = []
    for row in data:
        distance = [metric_function(row, centroids)]
        index = np.argmin(distance)
        assigned_cluster.append(index)
    return centroids, assigned_cluster


def plot_clusters(data, centroids, assigned_cluster, true_labels):
    sns.scatterplot(x=[x[0] for x in data],
                    y=[x[1] for x in data],
                    hue=true_labels,
                    style=assigned_cluster,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in centroids],
             [y for _, y in centroids],
             '+',
             markersize=10,
             )
    plt.show()
