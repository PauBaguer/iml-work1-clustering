from pam import *
from clara import *
from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import sklearn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


def random_neighbor(df, current):
    # we select a random medoid and a random nonmedoid to replace it
    non_medoids = df[~df.index.isin(current.index)]
    neighbor = non_medoids.sample(1)
    medoid = current.sample(1)
    current_tmp = current.copy()
    medoid_index = medoid.index.tolist()[0]
    medoid_id = current.index.get_loc(medoid_index)
    neighbor_index = neighbor.index.tolist()[0]
    current_tmp.loc[medoid_index] = df.loc[neighbor_index]
    current_tmp = current_tmp.rename(
        index={current_tmp.index[medoid_id]: neighbor_index})  # because when swapping rows, indeces do not swap as well
    return current_tmp, medoid, medoid_id, neighbor


def cost_differential(df, k, current, clusters, clusters2, S, new_clusters, new_clusters2, medoid, medoid_id, neighbor):
    Om = medoid  # current medoid that has been replaced
    Op = neighbor  # new medoid to replace Om
    medoids_tmp = S.copy()
    Cmp = 0
    for i in range(len(clusters[medoid_id])):
        Oj_idx = clusters[medoid_id][i][0]
        Oj = df.loc[Oj_idx]
        Oj2_idx = [element for element in clusters2 if element[0] == Oj_idx][0][1]
        Oj2 = current.iloc[Oj2_idx]

        # distance computations
        d_Oj_Oj2 = distance(Oj, Oj2)
        d_Oj_Om = distance(Oj, Om)
        d_Oj_Op = distance(Oj, Op)

        if d_Oj_Op >= d_Oj_Oj2:
            Cmp += (d_Oj_Oj2 - d_Oj_Om)
        elif d_Oj_Op < d_Oj_Oj2:
            Cmp += (d_Oj_Op - d_Oj_Om)

    for idx in clusters:
        if idx != medoid_id:
            for j in range(len(clusters[idx])):
                Oj_idx = clusters[idx][j][0]
                Oj = df.loc[Oj_idx]
                Oj2 = current.iloc[idx]

                # distances computations
                d_Oj_Oj2 = distance(Oj, Oj2)
                d_Oj_Om = distance(Oj, Om)
                d_Oj_Op = distance(Oj, Op)

                if d_Oj_Oj2 > d_Oj_Op:
                    Cmp += (d_Oj_Op - d_Oj_Oj2)
    return Cmp


def pre_validation(bestclusters):
    predictions = []
    for classe in range(len(bestclusters)):
        for i in range(len(bestclusters[classe])):
            predictions.append([bestclusters[classe][i][0], classe])
    sorted_predictions = sorted(predictions, key=lambda x: x[0])
    predictions_2 = [datapoint[1] for datapoint in sorted_predictions]
    predictions_array = np.array(predictions_2)
    return predictions_array


def trying_different_values(numlocal, maxneighbor, k, df, gt):
    results = []
    for num in numlocal:
        for maxn in maxneighbor:
            for k_val in k:
                bestnode, bestclusters = CLARANS(k_val, num, maxn, df)
                predictions_array = pre_validation(bestclusters)
                accuracy = metrics.accuracy_score(gt, predictions_array)
                results.append([[num, maxn, k_val, accuracy, [bestnode, bestclusters]]])
    return results


def CLARANS(k, numlocal, maxneighbor, df):
    print("Running CLARANS")

    mincost = math.inf
    i = 1
    bestnode = None

    while i <= numlocal:
        print('FINDING MINIMUM = ' + str(i))
        current = df.sample(n=k)
        [clusters, clusters2] = cluster_asignment(k, df, current)
        current_cost = compute_total_cost(k, clusters, current)
        print('Current cost = ' + str(current_cost))

        j = 1
        mincost_local = math.inf
        while j <= maxneighbor:
            print('Maxneighbor = ' + str(j))
            S, medoid, medoid_id, neighbor = random_neighbor(df, current)
            [new_clusters, new_clusters2] = cluster_asignment(k, df, S)
            cost_diff = cost_differential(df, k, current, clusters, clusters2, S, new_clusters, new_clusters2, medoid, medoid_id, neighbor)
            print('Cmp = ' + str(cost_diff))
            if cost_diff < 0:
                current = S
                # mincost_local_diff = cost_diff
                clusters = new_clusters
                clusters2 = new_clusters2
                j = 1
            else:
                j += 1
        mincost_local = compute_total_cost(k, clusters, current)
        if mincost_local < mincost:
            mincost = mincost_local
            bestnode = current

        if bestnode is None:  # if in this iteration no better node than random is found
            bestnode = current
        print('MinCost TOTAL = ' + str(mincost))

        i += 1

    bestclusters = cluster_asignment(k, df, bestnode)[0]

    return bestnode, bestclusters