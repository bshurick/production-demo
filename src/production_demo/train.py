import hashlib
import pandas as pd

from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from production_demo.constants import (CATEGORIES,
                                       NUMERICS,
                                       OUTPUT)


# custom transformer for categories
class CategoriesTransformer(BaseEstimator, TransformerMixin):
   
    @staticmethod
    def hash_col(x, n_buckets=100000):
        return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % n_buckets
    
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
    train = pd.read_csv('./data/train.csv')
    hct = CategoriesTransformer(CATEGORIES)
    model = Pipeline([
      ('hash', hct),
      ('LGBM', LGBMRegressor(random_state=22)),
    ])
    model.fit(train[NUMERICS], train[OUTPUT])
    dump(model, "./data/trained_model")
