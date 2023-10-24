from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from sklearn import metrics
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

    adult_df = load_arff('datasets/adult.arff')
    vowel_df = load_arff('datasets/vowel.arff')
    pen_based_df = load_arff('datasets/pen-based.arff')

    # Preprocessing for having categorical variables for K-Modes
    preprocessed_adult_df_cat, preprocessed_gs_adult_df_cat, preprocessor_pipeline_adult_cat = preprocessing.preprocess_df_to_categorical(adult_df)
    preprocessed_vowel_df_cat, preprocessed_gs_vowel_df_cat, preprocessor_pipeline_vowel_cat = preprocessing.preprocess_df_to_categorical(vowel_df)
    preprocessed_pen_df_cat, preprocessed_gs_pen_df_cat, preprocessor_pipeline_pen_cat = preprocessing.preprocess_df_to_categorical(pen_based_df)

    #####################################
    #                DBSCAN             #
    #####################################

    print("#####################################")
    print("#          DBSCAN adult df          #")
    print("#####################################")
    adult_dbscan_labels, adult_dbscan = dbscan.dbscan(preprocessed_adult_df, 1.6, 216, "euclidean", "auto")  # 60
    dbscan.plot_data(preprocessed_adult_df, adult_dbscan_labels, "Adult", "euclidean", "auto")

    dbscan.graph_dbscan_eps(preprocessed_adult_df, np.arange(1, 2, 0.1), 216, preprocessed_gs_adult_df, "euclidean", "Adult")
    print('Validation DBSCAN Adult df')
    validatordbscan = validation(dbscan.dbscan, preprocessed_adult_df, adult_dbscan_labels, 0, 0)
    validatordbscan.gold_standard_comparison(preprocessed_gs_adult_df)
    print()

    print("#####################################")
    print("#          DBSCAN vowel df          #")
    print("#####################################")
    vowel_dbscan_labels, vowel_dbscan = dbscan.dbscan(preprocessed_vowel_df, 0.85, 58, "euclidean", "auto")
    dbscan.plot_data(preprocessed_vowel_df, vowel_dbscan_labels, "Vowel", "euclidean", "auto")

    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.5, 1.1, 0.05), 58,preprocessed_gs_vowel_df, "euclidean", "Vowel")
    print('Validation DBSCAN Vowel df')
    validator_dbscan_vowel = validation(dbscan.dbscan, preprocessed_vowel_df, vowel_dbscan_labels, 2, 2)
    validator_dbscan_vowel.gold_standard_comparison(preprocessed_gs_vowel_df)
    print()


    print("#####################################")
    print("#          DBSCAN pen df            #")
    print("#####################################")

    print("Auto")
    pen_dbscan_labels, pen_dbscan_auto = dbscan.dbscan(preprocessed_pen_df, 0.44, 32, "euclidean", "auto") #0.415, 32
    dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean", "auto")
    silhouette_avg = metrics.silhouette_score(preprocessed_pen_df, pen_dbscan_labels)
    db_index = metrics.davies_bouldin_score(preprocessed_pen_df, pen_dbscan_labels)
    print(f"silhouette: {silhouette_avg}, db_index {db_index}")
    print('Validation DBSCAN Pen df')
    validatordbscan = validation(dbscan.dbscan, preprocessed_vowel_df, vowel_dbscan_labels, 0, 0)
    validatordbscan.gold_standard_comparison(preprocessed_gs_vowel_df)
    print()
    print("ball_tree")
    pen_dbscan_labels, pen_dbscan_balltree = dbscan.dbscan(preprocessed_pen_df, 0.44, 32, "euclidean", "ball_tree")  # 0.415, 32
    silhouette_avg = metrics.silhouette_score(preprocessed_pen_df, pen_dbscan_labels)
    db_index = metrics.davies_bouldin_score(preprocessed_pen_df, pen_dbscan_labels)
    print(f"silhouette: {silhouette_avg}, db_index {db_index}")

    print()
    print("kd_tree")
    pen_dbscan_labels, pen_dbscan_kdtree = dbscan.dbscan(preprocessed_pen_df, 0.44, 32, "euclidean", "kd_tree")  # 0.415, 32
    silhouette_avg = metrics.silhouette_score(preprocessed_pen_df, pen_dbscan_labels)
    db_index = metrics.davies_bouldin_score(preprocessed_pen_df, pen_dbscan_labels)
    print(f"silhouette: {silhouette_avg}, db_index {db_index}")
    dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean", "kd_tree")

    print()
    print("Brute")
    pen_dbscan_labels, pen_dbscan_brute = dbscan.dbscan(preprocessed_pen_df, 0.44, 32, "euclidean", "brute")  # 0.415, 32
    silhouette_avg = metrics.silhouette_score(preprocessed_pen_df, pen_dbscan_labels)
    db_index = metrics.davies_bouldin_score(preprocessed_pen_df, pen_dbscan_labels)
    print(f"silhouette: {silhouette_avg}, db_index {db_index}")


    pen_dbscan_labels, pen_dbscan = dbscan.dbscan(preprocessed_pen_df, 0.012, 32, "cosine", "auto")  # 0.415, 32
    dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "cosine", "auto")


    pen_dbscan_labels, pen_dbscan = dbscan.dbscan(preprocessed_pen_df, 1.25, 32, "manhattan", "auto")  # 0.415, 32
    dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "manhattan", "auto")

    print("EUCLIDEAN")
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.4, 0.5, 0.01), 32, preprocessed_gs_pen_df, "euclidean", "Pen-based")
    print("COSINE")
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.005, 0.02, 0.001), 32, preprocessed_gs_pen_df, "cosine", "Pen-based")
    print("MANHATTAN")
    dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(1.2, 1.5, 0.05), 32, preprocessed_gs_pen_df, "manhattan", "Pen-based")





    # ####################################
    #                Birch              #
    # ####################################

    print("#####################################")
    print("#           Birch adult df          #")
    print("#####################################")

    adult_birch_labels = birch.birch(preprocessed_adult_df, 2, 0.5)
    # birch.plot_data(preprocessed_adult_df, adult_birch_labels, "Adult")
    # birch.accuracy(preprocessed_gs_adult_df, adult_birch_labels)
    print('Validation Birch Adult df')
    validatorbirch = validation(birch.birch, preprocessed_adult_df, adult_birch_labels, 0, 0)
    validatorbirch.csearch(5, 'david bouldin score', 'Adult dataset')
    validatorbirch.csearch(5, 'silhouette score', 'Adult dataset')
    validatorbirch.gold_standard_comparison(preprocessed_gs_adult_df)
    print()
    print("#####################################")
    print("#           Birch vowel df          #")
    print("#####################################")
    vowel_birch_labels = birch.birch(preprocessed_vowel_df, 11, 0.5)#0.92
    # birch.plot_data(preprocessed_vowel_df, vowel_birch_labels, "Vowel")
    # birch.accuracy(preprocessed_gs_vowel_df, vowel_birch_labels)
    print('Validation Birch Vowel df')
    validatorbirch = validation(birch.birch, preprocessed_vowel_df, vowel_birch_labels, 0, 0)
    validatorbirch.csearch(15, 'david bouldin score', 'Vowel dataset')
    validatorbirch.csearch(15, 'silhouette score', 'Vowel dataset')
    validatorbirch.gold_standard_comparison(preprocessed_gs_vowel_df)
    print()

    print("#####################################")
    print("#           Birch pen df            #")
    print("#####################################")
    pen_birch_labels = birch.birch(preprocessed_pen_df, 10, 0.5)
    # birch.plot_data(preprocessed_pen_df, pen_birch_labels, "Pen-based")
    # birch.accuracy(preprocessed_gs_pen_df, pen_birch_labels)
    print('Validation Birch Pen df')
    validatorbirch = validation(birch.birch, preprocessed_pen_df, pen_birch_labels, 0, 0)
    validatorbirch.csearch(15, 'david bouldin score', 'Pen dataset')
    validatorbirch.csearch(15, 'silhouette score', 'Pen dataset')
    validatorbirch.gold_standard_comparison(preprocessed_gs_pen_df)
    print()

    print('#####################################')
    print('#              CLARANS              #')
    print('#####################################')

    # OPTIMAL PARAMETERS
    print('Parameter optimization Vowel dataset')
    preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    result1 = clarans.trying_different_values([1, 2, 3], [100], [11], preproc_vowel_df, preprocessed_gs_vowel_df)
    silhouette1, db1 = clarans.plot_test_cases_results(preproc_vowel_df, preprocessed_vowel_df,
                                                       preprocessed_gs_vowel_df, result1)
    print(f'Silhouette = {silhouette1}; DB = {db1}')

    preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    result2 = clarans.trying_different_values([2], [10, 100], [11], preproc_vowel_df, preprocessed_gs_vowel_df)
    silhouette2, db2 = clarans.plot_test_cases_results(preproc_vowel_df, preprocessed_vowel_df,
                                                       preprocessed_gs_vowel_df, result2)
    print(f'Silhouette = {silhouette2}; DB = {db2}')

    preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    result3 = clarans.trying_different_values([2], [10], [10, 11, 12, 13, 14], preproc_vowel_df,
                                              preprocessed_gs_vowel_df)
    silhouette3, db3 = clarans.plot_test_cases_results(preproc_vowel_df, preprocessed_vowel_df,
                                                       preprocessed_gs_vowel_df, result3, False, True)
    print(f'Silhouette = {silhouette3}; DB = {db3}')

    print('Parameter optimization Adult dataset')
    preproc_adult_df = pd.DataFrame(preprocessed_adult_df)
    result4 = clarans.trying_different_values([2], [10], [2, 3, 4], preproc_adult_df, preprocessed_gs_adult_df)
    silhouette4, db4 = clarans.plot_test_cases_results(preproc_adult_df, preprocessed_adult_df,
                                                       preprocessed_gs_adult_df, result4, False, True)
    print(f'Silhouette = {silhouette4}; DB = {db4}')

    print('Parameter optimization Pen-based dataset')
    preproc_pen_df = pd.DataFrame(preprocessed_pen_df)
    result5 = clarans.trying_different_values([2], [10], [9, 10, 11], preproc_pen_df, preprocessed_gs_pen_df)
    silhouette5, db5 = clarans.plot_test_cases_results(preproc_pen_df, preprocessed_pen_df, preprocessed_gs_pen_df,
                                                       result5, False, True)
    print(f'Silhouette = {silhouette5}; DB = {db5}')

    print('Adult dataset')
    preproc_adult_df = pd.DataFrame(preprocessed_adult_df)
    bestnode_adult, bestclusters_adult = clarans.CLARANS(preproc_adult_df, 2, 2, 10)
    pred_adult = clarans.pre_validation(bestclusters_adult)
    validatorclarans_adult = validation(clarans.CLARANS, preprocessed_adult_df, pred_adult, 2, 2)
    validatorclarans_adult.gold_standard_comparison(preprocessed_gs_adult_df)

    print('Vowel dataset')
    preproc_vowel_df = pd.DataFrame(preprocessed_vowel_df)
    bestnode_vowel, bestclusters_vowel = clarans.CLARANS(preproc_vowel_df, 11, 2, 10)
    pred_vowel = clarans.pre_validation(bestclusters_vowel)
    validatorclarans_vowel = validation(clarans.CLARANS, preprocessed_vowel_df, pred_vowel, 11, 11)
    validatorclarans_vowel.gold_standard_comparison(preprocessed_gs_vowel_df)

    print('Pen dataset')
    preproc_pen_df = pd.DataFrame(preprocessed_pen_df)
    bestnode_pen, bestclusters_pen = clarans.CLARANS(preproc_pen_df, 10, 2, 10)
    pred_pen = clarans.pre_validation(bestclusters_pen)
    validatorclarans_pen = validation(clarans.CLARANS, preprocessed_pen_df, pred_pen, 10, 10)
    validatorclarans_pen.gold_standard_comparison(preprocessed_gs_pen_df)

    print('#####################################')
    print('#           Fuzzy Cmeans            #')
    print('#####################################')

    print("#####################################")
    print("#             FCM vowel df          #")
    print("#####################################")

    mrange = [1.1, 1.6, 2, 2.6, 3]
    n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    best_m_SC, u_SC, v_SC, d_SC = fcmeans.msearch(preprocessed_vowel_df, mrange, n_clusters, 'silhouette score', False,
                                                  max_iter=10000, error_threshold=1e-4, metric='euclidean', v0=None)
    best_m_DBI, u_DBI, v_DBI, d_DBI = fcmeans.msearch(preprocessed_vowel_df, mrange, n_clusters, 'david bouldin score',
                                                      False, max_iter=10000, error_threshold=1e-4, metric='euclidean',
                                                      v0=None)
    print('best m SC: ', best_m_SC)
    print('best m DBI: ', best_m_DBI)

    validatorfcm = validation(fcmeans.fcm, preprocessed_vowel_df, u_SC.argmax(axis=1), 0, 0)
    validatorfcm.csearch(15, 'david bouldin score', 'Vowel dataset')
    validatorfcm.csearch(15, 'silhouette score', 'Vowel dataset')
    validatorfcm.gold_standard_comparison(preprocessed_gs_vowel_df)

    print("#####################################")
    print("#             FCM adult df          #")
    print("#####################################")

    mrange = [1.1, 1.6, 2, 2.6, 3]
    n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    best_m_SC, u_SC, v_SC, d_SC = fcmeans.msearch(preprocessed_adult_df, mrange, n_clusters, 'silhouette score', True,
                                                  max_iter=10000, error_threshold=1e-4, metric='euclidean', v0=None)
    best_m_DBI, u_DBI, v_DBI, d_DBI = fcmeans.msearch(preprocessed_adult_df, mrange, n_clusters, 'david bouldin score',
                                                      True, max_iter=10000, error_threshold=1e-4, metric='euclidean',
                                                      v0=None)
    print('best m SC: ', best_m_SC)
    print('best m DBI: ', best_m_DBI)

    validatorfcm_SC = validation(fcmeans.fcm, preprocessed_adult_df, u_SC.argmax(axis=1), 0, 0)
    validatorfcm_SC.csearch(5, 'david bouldin score', 'Adult dataset')
    validatorfcm_SC.csearch(5, 'silhouette score', 'Adult dataset')
    validatorfcm_SC.gold_standard_comparison(preprocessed_gs_adult_df)

    print("#####################################")
    print("#             FCM pen df          #")
    print("#####################################")

    mrange = [1.1, 1.6, 2, 2.6, 3]
    n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    best_m_SC, u_SC, v_SC, d_SC = fcmeans.msearch(preprocessed_pen_df, mrange, n_clusters, 'silhouette score', True,
                                                  max_iter=10000, error_threshold=1e-4, metric='euclidean', v0=None)
    best_m_DBI, u_DBI, v_DBI, d_DBI = fcmeans.msearch(preprocessed_pen_df, mrange, n_clusters, 'david bouldin score',
                                                      True, max_iter=10000, error_threshold=1e-4, metric='euclidean',
                                                      v0=None)
    print('best m SC: ', best_m_SC)
    print('best m DBI: ', best_m_DBI)

    validatorfcm_SC = validation(fcmeans.fcm, preprocessed_pen_df, u_SC.argmax(axis=1), 0, 0)
    validatorfcm_SC.csearch(15, 'david bouldin score', 'Pen dataset')
    validatorfcm_SC.csearch(15, 'silhouette score', 'Pen dataset')
    validatorfcm_SC.gold_standard_comparison(preprocessed_gs_pen_df)
    print()
    print('#####################################')
    print('#              K-Means              #')
    print('#####################################')
    n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    X_pen = preprocessed_pen_df
    centroid_pen, cluster_pen = kmeans.kmeans(X_pen, n_clusters)

    validatorkmeans = validation(kmeans.kmeans, preprocessed_pen_df, cluster_pen, centroid_pen, n_clusters)
    print('Pen dataset')
    validatorkmeans.gold_standard_comparison(preprocessed_gs_pen_df)
    kmeans.plot_kmeans_graphs(X_pen, "Pen-based", n_clusters)

    n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    X_vowel = preprocessed_vowel_df
    centroid_vowel, cluster_vowel = kmeans.kmeans(X_vowel, n_clusters)

    validatorkmeans = validation(kmeans.kmeans, preprocessed_vowel_df, cluster_vowel, centroid_vowel, n_clusters)
    print('Vowel dataset')
    validatorkmeans.gold_standard_comparison(preprocessed_gs_vowel_df)
    kmeans.plot_kmeans_graphs(X_vowel, "Vowel", n_clusters)

    n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    X_adult = preprocessed_adult_df
    centroid_adult, cluster_adult = kmeans.kmeans(X_adult, n_clusters)

    validatorkmeans = validation(kmeans.kmeans, preprocessed_adult_df, cluster_adult, centroid_adult, n_clusters)
    print('Adult dataset')
    validatorkmeans.gold_standard_comparison(preprocessed_gs_adult_df)

    kmeans.plot_kmeans_graphs(X_adult, "Adult", n_clusters)

    print('#####################################')
    print('#              K-Modes              #')
    print('#####################################')
    n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    X_pen = preprocessed_pen_df_cat
    centroid_pen, cluster_pen = kmodes.kmodes(X_pen, n_clusters)

    validatorkmodes = validation(kmodes.kmodes, preprocessed_pen_df_cat, cluster_pen, centroid_pen, n_clusters)
    print('Pen dataset')
    validatorkmodes.gold_standard_comparison(preprocessed_gs_pen_df_cat)
    kmodes.plot_kmodes_graphs(X_pen, "Pen-based", n_clusters)

    n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    X_vowel = preprocessed_vowel_df_cat
    centroid_vowel, cluster_vowel = kmodes.kmodes(X_vowel, n_clusters)

    validatorkmodes = validation(kmodes.kmodes, preprocessed_vowel_df_cat, cluster_vowel, centroid_vowel, n_clusters)
    print('Vowel dataset')
    validatorkmodes.gold_standard_comparison(preprocessed_gs_vowel_df_cat)
    kmodes.plot_kmodes_graphs(X_vowel, "Vowel", n_clusters)

    n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    X_adult = preprocessed_adult_df_cat
    centroid_adult, cluster_adult = kmodes.kmodes(X_adult, n_clusters)

    validatorkmodes = validation(kmodes.kmodes, preprocessed_adult_df_cat, cluster_adult, centroid_adult, n_clusters)
    print('Adult dataset')
    validatorkmodes.gold_standard_comparison(preprocessed_gs_adult_df_cat)

    kmodes.plot_kmodes_graphs(X_adult, "Adult", n_clusters)
