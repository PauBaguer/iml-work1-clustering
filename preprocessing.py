import numpy as np
from sklearn import preprocessing as pre
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#####################################
#             Missing values        #
#####################################

def interpolate_missing_values(df):

    return df

def erase_rows_with_missing_values(df):
    df_dropped_numerical = df.dropna(how='any')
    df_dropped_categorical = df_dropped_numerical
    for column in df_dropped_numerical:
        df_dropped_categorical = df_dropped_categorical.drop(df_dropped_categorical[df_dropped_categorical[column] == b'?'].index)

    return df_dropped_categorical



#####################################
#   Categorical to numerical        #
#####################################


def label_encoding(col):
    le = pre.LabelEncoder()
    le.fit(col)
    print(le.classes_)
    transform = le.transform(col)
    return transform,le

def label_decoding(col, le):
    return le.inverse_transform(col)

def one_hot_encoding(col):
    enc = pre.OneHotEncoder()
    enc.fit(col)
    encoded_arr = enc.transform(col).toarray()
    return encoded_arr,


#####################################
#     Normalization / scaling       #
#####################################

def standardization(col):
    return col

def mean_normalization(col):
    return col

def min_max_scaling(col):
    return col

def unit_vector(col):
    return col


#####################################
#   Main preprocessing function     #
#####################################
def preprocess_df(df):
    prepped_df = erase_rows_with_missing_values(df)



    categorical_cols = []
    numeric_cols = []
    for col in prepped_df:
        col_type = type(df[col].values[0])
        if col_type == bytes:
            categorical_cols.append(col)

            # Categorical data

            # Label encoder
            # print(df[col])
            # transform, label_encoder = label_encoding(df[col])
            # print(transform)

            # One hot encoding

        elif col_type == np.float64:
            # Numerical data
            numeric_cols.append(col)
        else:
            print('Check other type!!')

    # transform bytes to string.
    str_df = prepped_df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode("utf-8").unstack()
    for col in str_df:
        prepped_df[col] = str_df[col]


    # one hot encoder
    # categorical_df = preprocessed_df[categorical_cols]
    # categorical_str_df = categorical_df.stack().str.decode("utf-8").unstack()
    # transformed = one_hot_encoding(categorical_str_df)

    categorical_transformer = Pipeline(steps=[
        ("encoder", pre.OneHotEncoder())
    ])

    numeric_transformer = Pipeline(steps=[
        #("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(sparse_threshold=0, transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    preprocessor.fit(prepped_df)
    transformed_df = preprocessor.transform(prepped_df)
    print()

    # clustering = Pipeline(
    #     steps=[("preprocessor", preprocessor), ("clustering", DBSCAN(eps=0.3, min_samples=10))]
    # )
    # clustering.fit(prepped_df)

    X = transformed_df
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
            zorder=10
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()
    return preprocessor
