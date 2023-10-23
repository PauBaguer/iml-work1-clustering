from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn
import math

import preprocessing
import dbscan, birch




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