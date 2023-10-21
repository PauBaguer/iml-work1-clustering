import numpy as np
from sklearn import preprocessing as pre
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



#####################################
#             Missing values        #
#####################################

def erase_rows_with_missing_values(df):
    df_dropped_numerical = df.dropna(how='any')
    df_dropped_categorical = df_dropped_numerical
    for column in df_dropped_numerical:
        df_dropped_categorical = df_dropped_categorical.drop(df_dropped_categorical[df_dropped_categorical[column] == b'?'].index)

    return df_dropped_categorical



#####################################
#   Main preprocessing function     #
#####################################
def preprocess_df(df):
    prepped_df = df#erase_rows_with_missing_values(df)

    classification_goldstandard_cols = ["class", "a17", "Class"] # The columns to take out of preprocessing bc they are the final gold standard classification.
    goldstandard_col = []
    categorical_cols = []
    numeric_cols = []
    for col in prepped_df:
        if col in classification_goldstandard_cols:
            goldstandard_col.append(col)
            break
        col_type = type(df[col].values[0])
        if col_type == bytes:
            # Categorical data
            categorical_cols.append(col)

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

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), # fill missing values with most frequent
        ("encoder", pre.OneHotEncoder()),
        # ("scaler", StandardScaler(with_mean=False)),
        # ("min-max-scaler", MinMaxScaler())
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # fill missing values with the median
        ("scaler", StandardScaler()),
        ("min-max-scaler", MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(sparse_threshold=0, transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    preprocessor.fit(prepped_df)
    transformed_df = preprocessor.transform(prepped_df)
    print()

    goldstandard_preprocessor = pre.LabelEncoder()

    goldstandard_preprocessor.fit(prepped_df[goldstandard_col].values.ravel())
    transformed_goldstandard_col_df = goldstandard_preprocessor.transform(prepped_df[goldstandard_col].values.ravel())

    # clustering = Pipeline(
    #     steps=[("preprocessor", preprocessor), ("clustering", DBSCAN(eps=0.3, min_samples=10))]
    # )
    # clustering.fit(prepped_df)


    return transformed_df, transformed_goldstandard_col_df, preprocessor
