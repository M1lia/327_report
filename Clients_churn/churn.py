# TODO: take parameters from meta including type (b2b or b2c)
# TODO: grid search for hyperparameters tuning
# TODO: k-Fold Cross Validation

# TODO: normalize  data
# TODO: deal with lookup parameters
# TODO: write tests
# TODO: sklearn pipeline with catboost integration
# write custom catboost Classifier: https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7
# TODO: understand target_names_mapping


# structure of ModelSchemaMetadata in trainsession table in ml db
"""
{"inputs":[{"name":"Name","type":"Text","isRequired":true}],"output":{"name":"GenderId","type":"Lookup","displayName":"Gender"}}
"""
from sklearn_pandas import DataFrameMapper, gen_features
#from sklearn_pandas_transformers import SplitXY, EstimatorWithoutYWrapper, SklearnPandasWrapper
from typing import Tuple, List
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from app.common.logging import log
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

categorical_suffix = "_#CAT#"


class CustomFeatureSelection(SelectFromModel):

    def transform(self, X):
        # Get indices of important features
        important_features_indices = list(self.get_support(indices=True))

        # Select important features
        _X = X.iloc[:, important_features_indices].copy()

        return _X


class CustomCatBoostClassifier(CatBoostClassifier):
    """
    Custom classifier for determining catagorical features by suffix in
    their names
    example: name_of_column_#CAT#.

    categorical_suffix = "_#CAT#"

    """

    def fit(self, X, y=None, **fit_params):
        return super().fit(
            X,
            y=y,
            cat_features=list(X.filter(regex=f"{categorical_suffix}$").columns),
            **fit_params
        )


def __get_categorical_features(meta) -> Tuple[list, list, list]:
    """
    Returns a list of categorical features with added suffix
    :param: metadata
    :structure: {"inputs":[{"name":"Name","type":"Text","isRequired":true},{},{},{},{},{}]
                    "output":{"name":"GenderId","type":"Lookup","displayName":"Gender"}}
    :return: lists of cat, numeric and lookup features
    """
    categorical_features = []
    lookup_features = []
    numeric_features = []
    for i in range(len(meta['inputs'])):
        if meta['inputs'][i]['type'] == 'Text':
            # meta['inputs'][i]['name'] += categorical_suffix
            # categorical_features.append(str(meta['inputs'][i]['name']) + categorical_suffix)
            categorical_features.append(str(meta['inputs'][i]['name']))
        if meta['inputs'][i]['type'] == 'Lookup':
            lookup_features.append(meta['inputs'][i]['name'])
        if meta['inputs'][i]['type'] == 'Numeric':
            numeric_features.append(meta['inputs'][i]['name'])

    return categorical_features, lookup_features, numeric_features


def __rename_categorical_columns(categorical_features_names: list,
                                 categorical_suffix: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Renames categorical columns in dataframe
    :param: names of categorical features
    :param: categorical_suffix
    :return: dataframe with renamed columns
    """
    for i in range(len(dataframe.columns)):
        if dataframe.columns[i] in categorical_features_names:
            dataframe.rename(columns={str(dataframe.columns[i]): str(dataframe.columns[i]) + categorical_suffix},
                             inplace=True)
    return dataframe


def train_clients_churn(data_train, data_test, y_train, y_test, meta, target_names_mapping) -> Tuple[Pipeline, dict]:
    categorical_features = []
    lookup_features = []
    numeric_features = []
    categorical_features, lookup_features, numeric_features = __get_categorical_features(meta)
    data_train = __rename_categorical_columns(categorical_features, categorical_suffix, data_train)
    data_test = __rename_categorical_columns(categorical_features, categorical_suffix, data_test)

    model_summary = {'TrainSetSize': len(data_train)}
    model_summary.update({'TestSetSize': len(data_test)})

    log(f'Model summary = {model_summary}')
    class_negative = y_train.value_counts()[1]
    class_positive = y_train.value_counts()[0]

    model = CustomCatBoostClassifier(
        iterations=100,
        learning_rate=0.001,
        depth=2,
        verbose=False,
        class_weights=[1, (class_negative / class_positive)]
    )
    numeric_transformer = gen_features(
        columns=numeric_features,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "median"
            },
            {
                "class": StandardScaler
            }
        ]
    )

    preprocess_mapper = DataFrameMapper(
        [
            numeric_transformer
        ],
        input_df=True,
        df_out=True
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess_mapper),
        ("feature_selection", CustomFeatureSelection(CustomCatBoostClassifier(logging_level="Silent"))),
        ("estimator", CustomCatBoostClassifier(logging_level="Silent"))
    ])
    # numeric_transformer = Pipeline(
    #     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
    #            ]
    # )

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", numeric_transformer, numeric_features)
    #     ]
    # )




    # pipe = Pipeline(steps=[("preprocessor", preprocess_mapper), ("feature_selection",
    #                                                              CustomFeatureSelection(
    #                                                                  CustomCatBoostClassifier(logging_level="Silent"))),
    #                        ("CB_model", model)])

    pipeline.fit(data_train, y_train)
    predictions = pipeline.predict(data_test)
    score = roc_auc_score(y_test, predictions)

    model_summary.update({'Roc_auc_score': score})

    return model, model_summary


# TEST


path_to_data = r'C:\Users\Machine\Downloads\churn_daata\churn.csv'
df = pd.read_csv(path_to_data)
resulted_dataset = df[['avg_transaction_value', 'days_since_last_login', 'churn_risk_score',
                       'membership_category', 'age', 'gender']]
resulted_dataset.rename(columns={'churn_risk_score': 'is_churned'}, inplace=True)

train_data, test_data = train_test_split(resulted_dataset, test_size=0.2, random_state=42)
y_train = train_data['is_churned']
y_test = test_data['is_churned']
train_data = train_data.drop(['is_churned'], axis=1)
test_data = test_data.drop(['is_churned'], axis=1)
target_names_mapping = []

meta = {"inputs": [{"name": "avg_transaction_value", "type": "Numeric", "isRequired": 'true'},
                   {"name": "days_since_last_login", "type": "Numeric", "isRequired": 'true'},
                   {"name": "membership_category", "type": "Text", "isRequired": 'true'},
                   {"name": "age", "type": "Numeric", "isRequired": 'true'},
                   {"name": "gender", "type": "Text", "isRequired": 'true'}],
        "output": {"name": "GenderId", "type": "Lookup", "displayName": "Gender"}}

pipeline, summ = train_clients_churn(train_data, test_data, y_train, y_test, meta, target_names_mapping)
print(pipeline, summ)
