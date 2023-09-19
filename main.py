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
    heartc_df = load_arff('datasets/heart-c.arff')

    print(adult_df)
    print(heartc_df)
