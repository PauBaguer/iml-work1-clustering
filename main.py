from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import sklearn
import matplotlib.pyplot as plt
import seaborn
import math

import preprocessing
from preprocessing import *




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


    print(adult_df.shape)
    print(vowel_df.shape)
    print(pen_based_df.shape)


    #####################################
    #             Preprocessing         #
    #####################################
    preprocessing.preprocess_df(adult_df)
    preprocessing.preprocess_df(vowel_df)
    preprocessing.preprocess_df(pen_based_df)
    print()