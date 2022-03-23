import hashlib
import pandas as pd

from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from production_demo.constants import (CATEGORIES,
                                       MODEL_PARAMS,
                                       NUMERICS,
                                       OUTPUT)


class CategoriesTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for categories """

    @staticmethod
    def hash_col(x, n_buckets=100000):
        return int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16) % n_buckets
    
    def __init__(self, category_cols: list):
        self.category_cols = category_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        _X = X.copy()
        for c in self.category_cols:
            _X[c].fillna('', inplace=True)
            _X[c] = _X[c].apply(self.hash_col)
        return _X


def handler():
    """ Main entry point for model training """

    # load training data
    train = pd.read_csv('./data/train.csv')

    # create model artifacts 
    hct = CategoriesTransformer(CATEGORIES)
    model = Pipeline([
      ('hash', hct),
      ('LGBM', LGBMRegressor(**MODEL_PARAMS)),
    ])
    model.set_params(**{'LGBM__colsample_bytree': 0.4569053964451767, 
                        'LGBM__learning_rate': 0.060215823699935396, 
                        'LGBM__max_depth': 10, 
                        'LGBM__min_child_weight': 0.06494958890542044, 
                        'LGBM__min_split_gain': 0.05643039388402624, 
                        'LGBM__n_estimators': 354, 
                        'LGBM__num_leaves': 3192, 
                        'LGBM__reg_alpha': 260.8146685873852, 
                        'LGBM__reg_lambda': 101.43819465504578, 
                        'LGBM__subsample': 0.4382540932521588})

    # train model
    model.fit(train[NUMERICS + CATEGORIES], train[OUTPUT])

    # serialize trained model artifacts
    dump(model, "./data/trained_model")
