import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, silhouette_score

#####################################
#         Metric definitions        #
#####################################


# Compute the dissimilarity measure between a point and a vector of points.
# It is defined by the number of mismatches of the corresponding attribute
# categories of two objects.
def dissimilarity_measure(point, point_vector):
    return [np.sum(np.array(point) != np.array(p)) for p in point_vector]

# Compute the distance matrix to be used in the validation
def compute_distance_matrix(data):
    num_samples = data.shape[0]
    distance_matrix = [[] for _ in range(num_samples)]
    for i, row in enumerate(data):
        distance_matrix[i] = (dissimilarity_measure(row, data))
    return distance_matrix

#####################################
#           Initializations         #
#####################################


# Random initialization
def random_init(data, num_clusters):
    return data[np.random.choice(data.shape[0], num_clusters, replace=False)]


# Initialization according to Huang [1997]
def huang_init(data, num_clusters):
    return data[np.random.choice(data.shape[0], num_clusters, replace=False)]


initializations = {
    'random': random_init,
    'huang': huang_init,
}

#####################################
#              Clustering           #
#####################################


# K-modes implementation
def kmodes(data, num_clusters, max_iter=5000, mode_init='random'):
    init_function = initializations[mode_init]
    # Choose the initial centroids
    centroids = init_function(data, num_clusters)

    i = 0
    prev_centroids = None
    # Iterate until the clusters stabilize, or we reach the maximum of iterations
    while np.not_equal(centroids, prev_centroids).any() and i < max_iter:
        clusters = [[] for _ in range(num_clusters)]
        # Assign each point to the cluster with the closest centroid
        for row in data:
            distance = [dissimilarity_measure(row, centroids)]
            index = np.argmin(distance)
            clusters[index].append(row)

        # Assign the used centroids to the previous ones
        prev_centroids = centroids
        # Recompute the new centroids
        for i, cluster in enumerate(clusters):
            if np.any(cluster):
                mode = []
                for j in range(data.shape[1]):
                    cluster = np.array(cluster)
                    unique, count = np.unique(cluster[:, j], return_counts=True)
                    mode.append(unique[np.argmax(count)])
                centroids[i] = mode
        i += 1

    # Compute the final cluster assignation for the data
    assigned_cluster = []
    for row in data:
        distance = [dissimilarity_measure(row, centroids)]
        index = np.argmin(distance)
        assigned_cluster.append(index)
    return centroids, assigned_cluster

def plot_kmodes_graphs(df, dataset_name, real_k, range_k=4):
    n_clusters = []
    sil_scores = []

    distance_matrix = compute_distance_matrix(df)

    kmin = max(real_k - range_k, 2)
    kmax = real_k + range_k
    for k in range(kmin, kmax):
        n_clusters.append(k)

        centroid, labels = kmodes(df, k)
        sil = silhouette_score(distance_matrix, labels)
        sil_scores.append(sil)

    results_df = pd.DataFrame({"num_clusters": n_clusters, "SC": sil_scores})

    fig, ax = plt.subplots()
    ax.plot(results_df["num_clusters"], results_df["SC"], color='b')
    ax.legend()
    ax.set_title(f"K-Modes Silhouette, {dataset_name} dataset")
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette')
    fig.savefig(f"figures/kmodes/{dataset_name}-kmodes-silhouette.png")
    # fig.show()