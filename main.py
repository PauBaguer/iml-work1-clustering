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
    # adult_df = load_arff('datasets/adult.arff')
    #vowel_df = load_arff('datasets/vowel.arff')
    pen_based_df = load_arff('datasets/pen-based.arff')
    #breast_w_df = load_arff('datasets/breast-w.arff')


    #print(adult_df.shape)
    # print(vowel_df.shape)
    print(pen_based_df.shape)


    #####################################
    #             Preprocessing         #
    #####################################
    #preprocessed_adult_df, preprocessed_gs_adult_df, preprocessor_pipeline_adult = preprocessing.preprocess_df(adult_df)
    # preprocessed_vowel_df, preprocessed_gs_vowel_df, preprocessor_pipeline_vowel = preprocessing.preprocess_df(vowel_df)
    preprocessed_pen_df, preprocessed_gs_pen_df, preprocessor_pipeline_pen = preprocessing.preprocess_df(pen_based_df)
    print()
    # preprocessed_adult_df_dimensionality = preprocessed_adult_df.shape
    # print(f"preprocessed_adult_df_dimensionality: {preprocessed_adult_df_dimensionality}")

    # preprocessed_vowel_df_dimensionality = preprocessed_vowel_df.shape
    # print(f"preprocessed_vowel_df_dimensionality: {preprocessed_vowel_df_dimensionality}")

    preprocessed_pen_df_dimensionality = preprocessed_pen_df.shape
    print(f"preprocessed_pen_df_dimensionality: {preprocessed_pen_df_dimensionality}")
    #####################################
    #                DBSCAN             #
    #####################################

    # print("#####################################")
    # print("#          DBSCAN adult df          #")
    # print("#####################################")
    #dbscan.dbscan(preprocessed_adult_df, 1.5, 60)

    # print("#####################################")
    # print("#          DBSCAN vowel df          #")
    # print("#####################################")
    # dbscan.dbscan(preprocessed_vowel_df, 1.43, 15)

    print("#####################################")
    print("#          DBSCAN pen df            #")
    print("#####################################")

    # eps_arr = np.arange(0.4,0.5,0.01)
    # for eps in eps_arr:
    #     print(eps)
    #     dbscan.dbscan(preprocessed_pen_df, eps, 100)
    #dbscan.dbscan(preprocessed_pen_df, 0.42, 9)



    #####################################
    #                Birch              #
    #####################################

    # birch.birch(preprocessed_adult_df)
    # birch.birch(preprocessed_vowel_df)
    # birch.birch(preprocessed_pen_df)