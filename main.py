from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

import preprocessing
import dbscan, birch
import fcmeans, kmeans, pam, kmodes, clarans
from validation import validation
# import skfuzzy as fuzz

def load_arff(f_name):
    print(f'Opening, {f_name}')
    data, meta = arff.loadarff(f_name)
    df = pd.DataFrame(data)
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #####################################
    #             Load datasets         #
    #####################################
    adult_df = load_arff('datasets/adult.arff')
    vowel_df = load_arff('datasets/vowel.arff')
    pen_based_df = load_arff('datasets/pen-based.arff')


    #print(adult_df.shape)
    # print(vowel_df.shape)
    # print(pen_based_df.shape)


    #####################################
    #             Preprocessing         #
    #####################################
    preprocessed_adult_df, preprocessed_gs_adult_df, preprocessor_pipeline_adult = preprocessing.preprocess_df(adult_df, "Adult")
    preprocessed_vowel_df, preprocessed_gs_vowel_df, preprocessor_pipeline_vowel = preprocessing.preprocess_df(vowel_df, "Vowel")
    preprocessed_pen_df, preprocessed_gs_pen_df, preprocessor_pipeline_pen = preprocessing.preprocess_df(pen_based_df, "Pen-based")
    print()
    preprocessed_adult_df_dimensionality = preprocessed_adult_df.shape
    print(f"preprocessed_adult_df_dimensionality: {preprocessed_adult_df_dimensionality}")

    preprocessed_vowel_df_dimensionality = preprocessed_vowel_df.shape
    print(f"preprocessed_vowel_df_dimensionality: {preprocessed_vowel_df_dimensionality}")

    preprocessed_pen_df_dimensionality = preprocessed_pen_df.shape
    print(f"preprocessed_pen_df_dimensionality: {preprocessed_pen_df_dimensionality}")


    print("#####################################")
    print("#          DBSCAN adult df          #")
    print("#####################################")
    adult_dbscan_labels, adult_dbscan = dbscan.dbscan(preprocessed_adult_df, 1.6, 216, "euclidean", "auto")  # 60
    dbscan.plot_data(preprocessed_adult_df, adult_dbscan_labels, "Adult", "euclidean", "auto")
    dbscan.accuracy(preprocessed_gs_adult_df, adult_dbscan_labels)
    dbscan.graph_dbscan_eps(preprocessed_adult_df, np.arange(1, 2, 0.1), 216, preprocessed_gs_adult_df, "euclidean", "Adult")

    # print("#####################################")
    # print("#          DBSCAN vowel df          #")
    # print("#####################################")
    vowel_dbscan_labels, vowel_dbscan = dbscan.dbscan(preprocessed_vowel_df, 1.43, 58, "euclidean", "auto")
    dbscan.plot_data(preprocessed_vowel_df, vowel_dbscan_labels, "Vowel", "euclidean", "auto")
    dbscan.accuracy(preprocessed_gs_vowel_df, vowel_dbscan_labels)
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(1.2, 1.6, 0.02), 58,preprocessed_gs_vowel_df, "euclidean", "Vowel")
    #
    # validator_dbscan_vowel = validation(dbscan.dbscan, preprocessed_vowel_df, vowel_dbscan_labels,2, 2)
    # validator_dbscan_vowel.gold_standard_comparison(preprocessed_gs_vowel_df)
    #
    # print("#####################################")
    # print("#          DBSCAN pen df            #")
    # print("#####################################")
    #
    print("Auto")
    pen_dbscan_labels, pen_dbscan_auto = dbscan.dbscan(preprocessed_pen_df, 0.415, 32, "euclidean", "auto") #0.415, 32
    dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean", "auto")
    dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    # print()
    # print("ball_tree")
    # pen_dbscan_labels, pen_dbscan_balltree = dbscan.dbscan(preprocessed_pen_df, 0.415, 32, "euclidean", "ball_tree")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean", "ball_tree")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    # print()
    # print("kd_tree")
    # pen_dbscan_labels, pen_dbscan_kdtree = dbscan.dbscan(preprocessed_pen_df, 0.415, 32, "euclidean", "kd_tree")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean", "kd_tree")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    # print()
    # print("Brute")
    # pen_dbscan_labels, pen_dbscan_brute = dbscan.dbscan(preprocessed_pen_df, 0.415, 32, "euclidean", "brute")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean", "brute")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    #
    # pen_dbscan_labels, pen_dbscan = dbscan.dbscan(preprocessed_pen_df, 0.01, 32, "cosine", "auto")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "cosine", "auto")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    #
    # pen_dbscan_labels, pen_dbscan = dbscan.dbscan(preprocessed_pen_df, 0.95, 32, "manhattan", "auto")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "manhattan", "auto")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    # print("EUCLIDEAN")
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.4, 0.5, 0.01), 32, preprocessed_gs_pen_df, "euclidean", "Pen-based")
    # print("COSINE")
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.005, 0.02, 0.001), 32, preprocessed_gs_pen_df, "cosine", "Pen-based")
    # print("MANHATTAN")
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.9, 1.35, 0.01), 32, preprocessed_gs_pen_df, "manhattan", "Pen-based")



    # ####################################
    #                Birch              #
    # ####################################

    # print("#####################################")
    # print("#           Birch adult df          #")
    # print("#####################################")
    #
    # adult_birch_labels, adult_birch = birch.birch(preprocessed_adult_df, 0.5, 2)
    # birch.plot_data(preprocessed_adult_df, adult_birch_labels, "Adult")
    # birch.accuracy(preprocessed_gs_adult_df, adult_birch_labels)
    #
    # print("#####################################")
    # print("#           Birch vowel df          #")
    # print("#####################################")
    # vowel_birch_labels, vowel_birch = birch.birch(preprocessed_vowel_df, 0.5, 11)#0.92
    # birch.plot_data(preprocessed_vowel_df, vowel_birch_labels, "Vowel")
    # birch.accuracy(preprocessed_gs_vowel_df, vowel_birch_labels)
    #
    # print("#####################################")
    # print("#           Birch pen df            #")
    # print("#####################################")
    # pen_birch_labels, pen_birch = birch.birch(preprocessed_pen_df, 0.5, 10)
    # birch.plot_data(preprocessed_pen_df, pen_birch_labels, "Pen-based")
    # birch.accuracy(preprocessed_gs_pen_df, pen_birch_labels)
    #
    # print('#####################################')
    # print('#              CLARANS              #')
    # print('#####################################')
    #
    # # OPTIMAL PARAMETERS
    #
    # numlocal = [1, 2, 3]
    # maxneighbor = [100]
    # k = [11]
    # preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    # result1 = clarans.trying_different_values(numlocal, maxneighbor, k, preproc_vowel_df, preprocessed_gs_vowel_df)
    #
    # numlocal = [2]
    # maxneighbor = [100]
    # k = [10, 11, 12]
    # preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    # result2 = clarans.trying_different_values(numlocal, maxneighbor, k, preproc_vowel_df, preprocessed_gs_vowel_df)
    #
    # numlocal = [2]
    # maxneighbor = [10]
    # k = [11]
    # preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    # result3 = clarans.trying_different_values(numlocal, maxneighbor, k, preproc_vowel_df, preprocessed_gs_vowel_df)
    #
    # k = [2, 3, 4]
    # preproc_adult_df = pd.DataFrame(preprocessed_adult_df)
    # result4 = clarans.trying_different_values(numlocal, maxneighbor, k, preproc_adult_df, preprocessed_gs_adult_df)
    # silhouette, db = clarans.plot_test_cases_results(preproc_adult_df, preprocessed_adult_df, preprocessed_gs_adult_df,
    #                                                  result4)
    #
    # k = [9, 10, 11]
    # preproc_pen_df = pd.DataFrame(preprocessed_pen_df)
    # result5 = clarans.trying_different_values(numlocal, maxneighbor, k, preproc_pen_df, preprocessed_gs_pen_df)
    # silhouette, db = clarans.plot_test_cases_results(preproc_pen_df, preprocessed_pen_df, preprocessed_gs_pen_df,
    #                                                  result5)
    #
    # numlocal = 2
    # maxneighbor = 10
    #
    # k_adult = 2
    # k_vowel = 11
    # k_pen = 10
    #
    # print('Adult dataset')
    # preproc_adult_df = pd.DataFrame(preprocessed_adult_df)
    # bestnode_adult, bestclusters_adult = clarans.CLARANS(preproc_adult_df, k_adult, numlocal, maxneighbor)
    # pred_adult = clarans.pre_validation(bestclusters_adult)
    # validatorclarans_adult = validation(clarans.CLARANS, preprocessed_adult_df, pred_adult, k_adult, k_adult)
    # validatorclarans_adult.gold_standard_comparison(preprocessed_gs_adult_df)
    #
    # print('Vowel dataset')
    # preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    # bestnode_vowel, bestclusters_vowel = clarans.CLARANS(preproc_vowel_df, k_vowel, numlocal, maxneighbor)
    # pred_vowel = clarans.pre_validation(bestclusters_vowel)
    # validatorclarans_vowel = validation(clarans.CLARANS, preprocessed_vowel_df, pred_vowel, k_vowel, k_vowel)
    # validatorclarans_vowel.gold_standard_comparison(preprocessed_gs_vowel_df)
    #
    # print('Pen dataset')
    # preproc_pen_df = pd.DataFrame(preprocessed_pen_df)
    # bestnode_pen, bestclusters_pen = clarans.CLARANS(preproc_pen_df, k_pen, numlocal, maxneighbor)
    # pred_pen = clarans.pre_validation(bestclusters_pen)
    # validatorclarans_pen = validation(clarans.CLARANS, preprocessed_pen_df, pred_pen, k_pen, k_pen)
    # validatorclarans_pen.gold_standard_comparison(preprocessed_gs_pen_df)
    #
    # print('#####################################')
    # print('#           Fuzzy Kmeans            #')
    # print('#####################################')
    #
    # m = 2
    # n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    # X_vowel = preprocessed_vowel_df
    # uown_vowel, v_vowel, d_vowel = fcmeans.fcm(X_vowel, n_clusters, m, 10000)
    # #cntr_vowel, u_vowel, _, d_vowel, _, _, _ = fuzz.cluster.cmeans(X_vowel.T, n_clusters, m, error=1e-4, maxiter=10000)
    #
    # validatorfcm = validation(fcmeans.fcm, preprocessed_vowel_df, uown_vowel.argmax(axis=1), 0, 0)
    # validatorfcm.csearch(15, 'david bouldin score', 'Vowel dataset')
    # validatorfcm.csearch(15, 'silhouette score', 'Vowel dataset')
    # print('Vowel dataset', 'm = ', m)
    # # validatorfcm.library_comparison(u_vowel.argmax(axis=0))
    # validatorfcm.gold_standard_comparison(preprocessed_gs_vowel_df)
    # m=2
    # n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    # X_adult = preprocessed_adult_df
    # uown_adult, v_adult, d_adult = fcmeans.fcm(X_adult, n_clusters, m, 10000)
    # #cntr_adult, u_adult, _, d_adult, _, _, _ = fuzz.cluster.cmeans(X_adult.T, n_clusters, m, error=1e-4, maxiter=10000)
    #
    # validatorfcm = validation(fcmeans.fcm, preprocessed_adult_df, uown_adult.argmax(axis=1), 0, 0)
    # validatorfcm.csearch(5, 'david bouldin score', 'Adult dataset')
    # validatorfcm.csearch(5, 'silhouette score', 'Adult dataset')
    # print('Adult dataset', 'm = ', m)
    # # validatorfcm.library_comparison(u_adult.argmax(axis=0))
    # validatorfcm.gold_standard_comparison(preprocessed_gs_adult_df)
    #
    # m=1.1
    # n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    # X_pen = preprocessed_pen_df
    # uown_pen, v_pen, d_pen = fcmeans.fcm(X_pen, n_clusters, m, 10000)
    # #cntr_pen, u_pen, _, d_pen, _, _, _ = fuzz.cluster.cmeans(X_pen.T, n_clusters, m, error=1e-4, maxiter=10000)
    #
    # validatorfcm = validation(fcmeans.fcm, preprocessed_pen_df, uown_pen.argmax(axis=1), 0, 0)
    # validatorfcm.csearch(15, 'david bouldin score', 'Pen dataset')
    # validatorfcm.csearch(15, 'silhouette score', 'Pen dataset')
    # print('Pen dataset', 'm = ', m)
    # # validatorfcm.library_comparison(u_pen.argmax(axis=0))
    # validatorfcm.gold_standard_comparison(preprocessed_gs_pen_df)
    #
    # # difference_vowel = np.sqrt(np.sum(np.square(uown_vowel - u_vowel.T))/u_vowel.shape[0])
    # # difference_adult = np.sqrt(np.sum(np.square(uown_adult - u_adult.T))/u_adult.shape[0])
    # # difference_pen = np.sqrt(np.sum(np.square(uown_pen - u_pen.T))/u_pen.shape[0])
    # # print(f'Difference vowel U matix: {difference_vowel}')
    # # print(f'Difference adult U matix: {difference_adult}')
    # # print(f'Difference pen U matix: {difference_pen}')
    # print()
    #
    #
    # print('#####################################')
    # print('#              K-Means              #')
    # print('#####################################')
    # n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    # X_pen = preprocessed_pen_df
    # centroid_pen, cluster_pen = kmeans.kmeans(X_pen, n_clusters)
    #
    # validatorkmeans = validation(kmeans.kmeans, preprocessed_pen_df, cluster_pen, centroid_pen, n_clusters)
    # validatorkmeans.csearch(15, 'david bouldin score', 'Pen dataset')
    # validatorkmeans.csearch(15, 'silhouette score', 'Pen dataset')
    # print('Pen dataset')
    # validatorkmeans.library_comparison(cluster_pen)
    # validatorkmeans.gold_standard_comparison(preprocessed_gs_pen_df)
    #
    # n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    # X_vowel = preprocessed_vowel_df
    # centroid_vowel, cluster_vowel = kmeans.kmeans(X_vowel, n_clusters)
    #
    # validatorkmeans = validation(kmeans.kmeans, preprocessed_vowel_df, cluster_vowel, centroid_vowel, n_clusters)
    # validatorkmeans.csearch(15, 'david bouldin score', 'Vowel dataset')
    # validatorkmeans.csearch(15, 'silhouette score', 'Vowel dataset')
    # print('Vowel dataset')
    # validatorkmeans.library_comparison(cluster_vowel)
    # validatorkmeans.gold_standard_comparison(preprocessed_gs_vowel_df)
    #
    # n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    # X_adult = preprocessed_adult_df
    # centroid_adult, cluster_adult = kmeans.kmeans(X_adult, n_clusters)
    #
    # validatorkmeans = validation(kmeans.kmeans, preprocessed_adult_df, cluster_adult, centroid_adult, n_clusters)
    # validatorkmeans.csearch(5, 'david bouldin score', 'Adult dataset')
    # validatorkmeans.csearch(5, 'silhouette score', 'Adult dataset')
    # print('adult dataset')
    # validatorkmeans.library_comparison(cluster_adult)
    # validatorkmeans.gold_standard_comparison(preprocessed_gs_adult_df)
    #
    # print('#####################################')
    # print('#              K-Modes              #')
    # print('#####################################')
    # n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    # X_pen = preprocessed_pen_df
    # centroid_pen, cluster_pen = kmeans.kmeans(X_pen, n_clusters)
    #
    # validatorkmodes = validation(kmeans.kmeans, preprocessed_pen_df, cluster_pen, centroid_pen, n_clusters)
    # validatorkmodes.csearch(15, 'david bouldin score', 'Pen dataset')
    # validatorkmodes.csearch(15, 'silhouette score', 'Pen dataset')
    # print('Pen dataset')
    # validatorkmodes.library_comparison(cluster_pen)
    # validatorkmodes.gold_standard_comparison(preprocessed_gs_pen_df)
    #
    # n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    # X_vowel = preprocessed_vowel_df
    # centroid_vowel, cluster_vowel = kmeans.kmeans(X_vowel, n_clusters)
    #
    # validatorkmodes = validation(kmeans.kmeans, preprocessed_vowel_df, cluster_vowel, centroid_vowel, n_clusters)
    # validatorkmodes.csearch(15, 'david bouldin score', 'Vowel dataset')
    # validatorkmodes.csearch(15, 'silhouette score', 'Vowel dataset')
    # print('Vowel dataset')
    # validatorkmodes.library_comparison(cluster_vowel)
    # validatorkmodes.gold_standard_comparison(preprocessed_gs_vowel_df)
    #
    # n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    # X_adult = preprocessed_adult_df
    # centroid_adult, cluster_adult = kmeans.kmeans(X_adult, n_clusters)
    #
    # validatorkmodes = validation(kmeans.kmeans, preprocessed_adult_df, cluster_adult, centroid_adult, n_clusters)
    # validatorkmodes.csearch(5, 'david bouldin score', 'Adult dataset')
    # validatorkmodes.csearch(5, 'silhouette score', 'Adult dataset')
    # print('adult dataset')
    # validatorkmodes.library_comparison(cluster_adult)
    # validatorkmodes.gold_standard_comparison(preprocessed_gs_adult_df)