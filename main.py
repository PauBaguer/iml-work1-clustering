from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import sklearn
import matplotlib.pyplot as plt
import seaborn




def load_arff(f_name):
    print(f'Opening, {f_name}')
    data, meta = arff.loadarff(f_name)
    df = pd.DataFrame(data)
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    adult_df = load_arff('datasets/adult.arff')
    vowel_df = load_arff('datasets/vowel.arff')
    pen_based_df = load_arff('datasets/pen-based.arff')

    print(adult_df)
    print(vowel_df)
    print(pen_based_df)
