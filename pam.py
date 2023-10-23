from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import sklearn
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt


def distance(a, b):
    distance = np.linalg.norm(a - b)
    return distance

def compute_total_cost(k, clusters, medoids):
    total_cost = 0
    for i in range(k):
        for j in range(len(clusters[i])):
            if clusters[i][j][0] not in medoids.index:
                total_cost += clusters[i][j][1]
    return total_cost


def cluster_asignment(k, df, medoids):
    clusters = {}
    clusters2 = []
    for i in range(k):
        clusters[i] = []

    for index, data_point in df.iterrows():
        distances = np.linalg.norm(data_point - medoids, axis=1)  # euclidean distance
        min_indices = np.argsort(distances)[:2]
        min_values = [distances[i] for i in min_indices]
        clusters[min_indices[0]].append([index, min_values[0]])
        clusters2.append([index, min_indices[1]])
    return [clusters, clusters2]


def evaluate_new_medoids(k, df, clusters, medoids, current_cost, clusters2):
    best4cluster = {}
    for medoid in range(k):  # for each medoid (Om)
        print('Replacing medoid: ' + str(medoid))
        medoids_tmp = medoids.copy()
        min_cost = 0
        best_medoid_idx = None
        best_medoids = None
        best_clusters = None
        cost_total = math.inf
        for n in range(len(clusters[medoid])):  # for each non-medoid, we try it as new medoid (Op)

            if clusters[medoid][n][0] not in medoids.index:
                new_medoid_index = clusters[medoid][n][0]
                medoids_tmp.iloc[medoid] = df.loc[new_medoid_index]
                medoids_tmp = medoids_tmp.rename(index={medoids_tmp.index[medoid]: new_medoid_index})
                Om = medoids.iloc[medoid]
                Op = df.loc[new_medoid_index]
                Cmp = 0
                for i in range(len(clusters[medoid])):
                    Oj_idx = clusters[medoid][i][0]
                    Oj = df.loc[Oj_idx]
                    Oj2_idx = [element for element in clusters2 if element[0] == Oj_idx][0][1]
                    Oj2 = medoids.iloc[Oj2_idx]

                    d_Oj_Oj2 = distance(Oj, Oj2)
                    d_Oj_Om = distance(Oj, Om)
                    d_Oj_Op = distance(Oj, Op)

                    if d_Oj_Op >= d_Oj_Oj2:
                        Cmp += (d_Oj_Oj2 - d_Oj_Om)

                    elif d_Oj_Op < d_Oj_Oj2:
                        Cmp += (d_Oj_Op - d_Oj_Om)

                for idx in clusters:
                    if idx != medoid:
                        for j in range(len(clusters[idx])):
                            Oj_idx = clusters[idx][j][0]
                            Oj = df.loc[Oj_idx]
                            Oj2 = medoids.iloc[idx]

                            d_Oj_Oj2 = distance(Oj, Oj2)
                            d_Oj_Om = distance(Oj, Om)
                            d_Oj_Op = distance(Oj, Op)

                            if d_Oj_Oj2 > d_Oj_Op:
                                Cmp += (d_Oj_Op - d_Oj_Oj2)

                if Cmp < min_cost:
                    min_cost = Cmp
                    best_medoid_idx = new_medoid_index
                    best_medoids = medoids_tmp

        best4cluster[medoid] = [best_medoid_idx, min_cost, best_medoids]

    best_new_medoid = min(best4cluster, key=lambda key: best4cluster[key][1])
    new_cost = best4cluster[best_new_medoid][1]
    new_medoids = best4cluster[best_new_medoid][2]
    new_clusters, new_clusters2 = cluster_asignment(k, df, new_medoids)
    new_cost = compute_total_cost(k, new_clusters, new_medoids)

    if new_cost < current_cost:
        return [new_clusters, new_medoids, new_cost, new_clusters2]
    return None


def pam(df, k):
    medoids = df.sample(n=k)
    medoids_arr = medoids.to_numpy()
    clusters, clusters2 = cluster_asignment(k, df, medoids)
    current_cost = compute_total_cost(k, clusters, medoids)
    print("Current PAM cost = " + str(current_cost))
    result = [clusters, medoids, current_cost, clusters2]
    for iterations in range(10):
        result_before = result
        result = evaluate_new_medoids(k, df, result[0], result[1], result[2], result[3])
        if result is None:
            print('PAM has converged')
            return result_before
            break
        print("Current PAM cost = " + str(result[2]))
    print('Maximum number of iterations reached')
    return result