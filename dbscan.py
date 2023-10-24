import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

def dbscan(X, eps, min_samples, metric, algorithm):
    print(f"Params: eps: {eps}, min_samples: {min_samples}, metric: {metric}, algorithm: {algorithm}")
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(X)
    labels = db.labels_
    end = time.time()
    print(f"DBSCAN execution elapsed: {end - start}s")
    return labels, db


    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    #
    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)
    #
    # unique_labels = set(labels)
    # core_samples_mask = np.zeros_like(labels, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    #
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = labels == k
    #
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markersize=3,
    #         zorder=10
    #     )
    #
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markersize=1,
    #     )
    #
    # plt.title(f"DBSCAN. Nº clusters: {n_clusters_}")
    # plt.show()

def plot_data(X, labels, dataset_name, metric, algorithm):

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)


    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)


    print("Estimated number of points per cluster")
    points_per_cluster = {}
    unique_labels = set(labels)
    unique_labels_wo_noise_count = len(unique_labels)
    if -1 in unique_labels:
        unique_labels_wo_noise_count = unique_labels_wo_noise_count - 1


    for l in unique_labels:
        if l == -1:
            points_per_cluster[l] = list(labels).count(l)
            print(f"Noise points, {l}: {points_per_cluster[l]} points")
            break
        points_per_cluster[l] = list(labels).count(l)
        print(f"Cluster {l}: {points_per_cluster[l]} points")

    #colors = ["#8c510a","#d8b365","#f6e8c3","#c7eae5","#5ab4ac","#01665e","#e66101","#fdb863","#a6dba0","#008837", "red"]
    colors = cm.rainbow(np.linspace(0, 1, unique_labels_wo_noise_count))
   # color_labels = ['k' if i == -1 else i for i in labels]


    plt.scatter(X[:,0:1], X[:,1:2], c=labels, s=3)

    # i=0
    # for row in X:
    #     if labels[i] == -1:
    #         plt.plot(
    #             row[0],
    #             row[1],
    #             ".",
    #             color='k',
    #             markersize=2,
    #             zorder=-1
    #         )
    #     else:
    #         plt.plot(
    #             row[0],
    #             row[1],
    #             ".",
    #             color=colors[labels[i]],
    #             markersize=3,
    #             zorder=labels[i]
    #         )
    #     # if i > 100:
    #     #     plt.show()
    #     #     return
    #     i = i+1




    plt.title(f"DBSCAN {dataset_name}, metric: {metric}, algorithm {algorithm}. Nº clusters: {n_clusters_}")
    plt.savefig(f"figures/dbscan/{dataset_name}-graph.png")
    plt.show()

def graph_dbscan_eps(df, eps_range, min_samples,gs, metric, dataset_name):

    n_clusters_arr = []
    n_noise_arr = []
    accuracy_arr = []
    for eps in eps_range:
         print(eps)
         labels, db = dbscan(df, eps, min_samples, metric, "auto")
         acc = accuracy(gs, labels)
         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
         n_noise_ = list(labels).count(-1)
         n_clusters_arr.append(n_clusters_)
         n_noise_arr.append(n_noise_)
         accuracy_arr.append(acc)


    results_df = pd.DataFrame({"eps": eps_range, "n_clusters":n_clusters_arr, "n_noise":n_noise_arr, "accuracy": accuracy_arr})

    plt.plot(results_df["eps"], results_df["n_clusters"], marker='x')
    plt.title(f"DBSCAN, metric: {metric}, nº of clusters vs Epsilon")
    plt.grid()
    plt.savefig(f"figures/dbscan/{dataset_name}-{metric}-nclusters.png")
    plt.show()

    plt.plot(results_df["eps"], results_df["accuracy"], marker='x')
    plt.title(f"DBSCAN {dataset_name}, metric: {metric}, accuracy vs Epsilon")
    plt.grid()
    plt.savefig(f"figures/dbscan/{dataset_name}-{metric}-accuracy.png")
    plt.show()

def accuracy(gs, labels):
    count = 0
    for idx, x in enumerate(gs):
        if x == labels[idx]:
            count = count + 1

    acc = count / len(gs)
    print(f'Count {count}')
    print(f'Accuracy {acc}')
    return acc


