from pam import *
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
    #medoids_tmp = S.copy()
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
                bestnode, bestclusters = CLARANS(df, k_val, num, maxn)
                predictions_array = pre_validation(bestclusters)
                results.append([num, maxn, k_val, [bestnode, bestclusters]])
    return results


def plot_test_cases_results(preproc_df, preprocessed_df, preprocessed_gs, result, plot_vars=False, plot_measures=False):
    new_series = pd.Series(preprocessed_gs, name='GT')
    df = pd.concat([preproc_df, new_series], axis=1)
    silhouette = {}
    db = {}
    for testcase in range(len(result)):
        bestnode = result[testcase][3][0]
        numlocal = result[testcase][0]
        maxneighbor = result[testcase][1]
        k = result[testcase][2]

        bestnode_arr = bestnode.to_numpy()
        if plot_vars:
            for i in range(len(preprocessed_df[0]) - 1):
                plt.figure(figsize=(8, 6))
                plt.scatter(df[i], df[i + 1], c=df.iloc[:, -1], cmap='Pastel1', s=20)
                plt.scatter(bestnode_arr[:, i], bestnode_arr[:, i + 1], color='#000000')
                plt.title(f'K = {k}; numlocal = {numlocal}, maxneighbor = {maxneighbor}, Variables {i} {i + 1}')
                plt.show()

        bestclusters = result[testcase][3][1]
        predictions_array = pre_validation(bestclusters)
        silhouette_avg = metrics.silhouette_score(preprocessed_df, predictions_array)
        silhouette[testcase] = silhouette_avg
        db_index = metrics.davies_bouldin_score(preprocessed_df, predictions_array)
        db[testcase] = db_index
    if plot_measures:
        ks = [result[x][2] for x in range(len(result))]
        y_sil = list(silhouette.values())
        plt.figure()
        plt.plot(ks, y_sil)
        plt.xlabel('k values')
        plt.ylabel('Silhouette')
        plt.show()
        y_db = list(db.values())
        plt.figure()
        plt.plot(ks, y_db)
        plt.xlabel('k values')
        plt.ylabel('DB')
        plt.show()
    return silhouette, db


def CLARANS(df,k, numlocal = 2, maxneighbor = 100):
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