from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

import preprocessing
import dbscan, birch
import fcmeans, kmeans, pam, kmodes
from validation import validation
import skfuzzy as fuzz

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
    #breast_w_df = load_arff('datasets/breast-w.arff')


    #print(adult_df.shape)
    # print(vowel_df.shape)
    # print(pen_based_df.shape)


    #####################################
    #             Preprocessing         #
    #####################################
    preprocessed_adult_df, preprocessed_gs_adult_df, preprocessor_pipeline_adult = preprocessing.preprocess_df(adult_df)
    preprocessed_vowel_df, preprocessed_gs_vowel_df, preprocessor_pipeline_vowel = preprocessing.preprocess_df(vowel_df)
    preprocessed_pen_df, preprocessed_gs_pen_df, preprocessor_pipeline_pen = preprocessing.preprocess_df(pen_based_df)
    print()
    preprocessed_adult_df_dimensionality = preprocessed_adult_df.shape
    print(f"preprocessed_adult_df_dimensionality: {preprocessed_adult_df_dimensionality}")

    preprocessed_vowel_df_dimensionality = preprocessed_vowel_df.shape
    print(f"preprocessed_vowel_df_dimensionality: {preprocessed_vowel_df_dimensionality}")

    preprocessed_pen_df_dimensionality = preprocessed_pen_df.shape
    print(f"preprocessed_pen_df_dimensionality: {preprocessed_pen_df_dimensionality}")
    #####################################
    #                DBSCAN             #
    #####################################

    print("#####################################")
    print("#          DBSCAN adult df          #")
    print("#####################################")
    # adult_dbscan_labels = dbscan.dbscan(preprocessed_adult_df, 1.6, 216, "euclidean") # 60
    # dbscan.plot_data(preprocessed_adult_df, adult_dbscan_labels, "Adult", "euclidean")
    # dbscan.accuracy(preprocessed_gs_adult_df, adult_dbscan_labels)
    # dbscan.graph_dbscan_eps(preprocessed_adult_df, np.arange(1, 2, 0.1))
    # 
    # print("#####################################")
    # print("#          DBSCAN vowel df          #")
    # print("#####################################")
    # vowel_dbscan_labels = dbscan.dbscan(preprocessed_vowel_df, 1.43, 58, "euclidean")
    # dbscan.plot_data(preprocessed_vowel_df, vowel_dbscan_labels, "Vowel", "euclidean")
    # dbscan.accuracy(preprocessed_gs_vowel_df, vowel_dbscan_labels)
    # dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(1.2, 1.6, 0.02))
    # 
    # print("#####################################")
    # print("#          DBSCAN pen df            #")
    # print("#####################################")
    # 
    # pen_dbscan_labels = dbscan.dbscan(preprocessed_pen_df, 0.415, 32, "euclidean") #0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "euclidean")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    #
    # pen_dbscan_labels = dbscan.dbscan(preprocessed_pen_df, 0.01, 32, "cosine")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "cosine")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    #
    # pen_dbscan_labels = dbscan.dbscan(preprocessed_pen_df, 0.95, 32, "manhattan")  # 0.415, 32
    # dbscan.plot_data(preprocessed_pen_df, pen_dbscan_labels, "Pen", "manhattan")
    # dbscan.accuracy(preprocessed_gs_pen_df, pen_dbscan_labels)
    # print("EUCLIDEAN")
    # dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.4, 0.5, 0.01), preprocessed_gs_pen_df, "euclidean")
    # print("COSINE")
    # dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.005, 0.02, 0.001), preprocessed_gs_pen_df, "cosine")
    # print("MANHATTAN")
    # dbscan.graph_dbscan_eps(preprocessed_pen_df, np.arange(0.9, 1.35, 0.01), preprocessed_gs_pen_df, "manhattan")



    #####################################
    #                Birch              #
    #####################################

    # print("#####################################")
    # print("#           Birch adult df          #")
    # print("#####################################")
    #
    # adult_birch_labels = birch.birch(preprocessed_adult_df, 0.5, 2)
    # birch.plot_data(preprocessed_adult_df, adult_birch_labels)
    # birch.accuracy(preprocessed_gs_adult_df, adult_birch_labels)
    #
    # print("#####################################")
    # print("#           Birch vowel df          #")
    # print("#####################################")
    # vowel_birch_labels = birch.birch(preprocessed_vowel_df, 0.5, 11)#0.92
    # birch.plot_data(preprocessed_vowel_df, vowel_birch_labels)
    # birch.accuracy(preprocessed_gs_vowel_df, vowel_birch_labels)
    #
    # print("#####################################")
    # print("#           Birch pen df            #")
    # print("#####################################")
    # pen_birch_labels = birch.birch(preprocessed_pen_df, 0.5, 10)
    # birch.plot_data(preprocessed_pen_df, pen_birch_labels)
    # birch.accuracy(preprocessed_gs_pen_df, pen_birch_labels)
    
    print('#####################################')
    print('#           Fuzzy Kmeans            #')
    print('#####################################')    
    
    m = 2
    n_clusters = preprocessed_gs_vowel_df[preprocessed_gs_vowel_df.argmax()] + 1
    X_vowel = preprocessed_vowel_df
    uown_vowel, v_vowel, d_vowel = fcmeans.fcm(X_vowel, n_clusters, m, 10000)
    cntr_vowel, u_vowel, _, d_vowel, _, _, _ = fuzz.cluster.cmeans(X_vowel.T, n_clusters, m, error=1e-4, maxiter=10000)
    
    validatorfcm = validation(fcmeans.fcm, preprocessed_vowel_df, uown_vowel.argmax(axis=1), 0, 0)
    validatorfcm.csearch(5, 'david bouldin score')
    validatorfcm.csearch(5, 'silhouette score')
    print('Vowel dataset')
    validatorfcm.library_comparison(u_vowel.argmax(axis=0))
    validatorfcm.gold_standard_comparison(preprocessed_gs_vowel_df)

    n_clusters = preprocessed_gs_adult_df[preprocessed_gs_adult_df.argmax()] + 1
    X_adult = preprocessed_adult_df
    uown_adult, v_adult, d_adult = fcmeans.fcm(X_adult, n_clusters, m, 10000)
    cntr_adult, u_adult, _, d_adult, _, _, _ = fuzz.cluster.cmeans(X_adult.T, n_clusters, m, error=1e-4, maxiter=10000)
    
    validatorfcm = validation(fcmeans.fcm, preprocessed_adult_df, uown_adult.argmax(axis=1), 0, 0)
    validatorfcm.csearch(5, 'david bouldin score')
    validatorfcm.csearch(5, 'silhouette score')
    print('Adult dataset')
    validatorfcm.library_comparison(u_adult.argmax(axis=0))
    validatorfcm.gold_standard_comparison(preprocessed_gs_adult_df)

    n_clusters = preprocessed_gs_pen_df[preprocessed_gs_pen_df.argmax()] + 1
    X_pen = preprocessed_pen_df
    uown_pen, v_pen, d_pen = fcmeans.fcm(X_pen, n_clusters, m, 10000)
    cntr_pen, u_pen, _, d_pen, _, _, _ = fuzz.cluster.cmeans(X_pen.T, n_clusters, m, error=1e-4, maxiter=10000)
    
    validatorfcm = validation(fcmeans.fcm, preprocessed_pen_df, uown_pen.argmax(axis=1), 0, 0)
    validatorfcm.csearch(5, 'david bouldin score')
    validatorfcm.csearch(5, 'silhouette score')
    print('Pen dataset')
    validatorfcm.library_comparison(u_pen.argmax(axis=0))
    validatorfcm.gold_standard_comparison(preprocessed_gs_pen_df)

    difference_vowel = np.sqrt(np.sum(np.square(uown_vowel - u_vowel.T))/u_vowel.shape[0])
    difference_adult = np.sqrt(np.sum(np.square(uown_adult - u_adult.T))/u_adult.shape[0])
    difference_pen = np.sqrt(np.sum(np.square(uown_pen - u_pen.T))/u_pen.shape[0])
    print(f'Difference vowel U matix: {difference_vowel}')
    print(f'Difference adult U matix: {difference_adult}')
    print(f'Difference pen U matix: {difference_pen}')
    print()
    