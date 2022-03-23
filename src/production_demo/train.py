"""Training module

This module trains contains a command-line entrypoint to
train and save model from a data/train.csv file.
"""

import hashlib
import logging
import pandas as pd

from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from production_demo.constants import CATEGORIES, MODEL_PARAMS, NUMERICS, OUTPUT


# set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)-15s: %(funcName)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


class CategoriesTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for categories"""

    @staticmethod
    def hash_col(x, n_buckets=100000):
        """Hash a string value into numeric buckets"""
        return int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % n_buckets

    def __init__(self, category_cols: list, n_buckets: int = 100000):
        """Choose columns to hash into N buckets"""
        self.category_cols = category_cols
        self.n_buckets = n_buckets

    def fit(self, X, y=None):
        """Implements a fit function but does nothing"""
        # no fit; return self
        return self

    def transform(self, X):
        """Apply our hashing function to categorical columns"""
        _X = X.copy()
        for c in self.category_cols:
            _X[c].fillna("", inplace=True)
            _X[c] = _X[c].apply(lambda x: self.hash_col(x, self.n_buckets))
        return _X


def handler():
    """Main entry point for model training"""

    # Load training data
    logger.info("Loading data")
    train = pd.read_csv("./data/train.csv")

    # Create model artifacts
    logger.info("Set up training pipeline")
    hct = CategoriesTransformer(CATEGORIES)
    model = Pipeline(
        [
            ("hash", hct),
            ("LGBM", LGBMRegressor()),
        ]
    )
    model.set_params(**MODEL_PARAMS)

    # Train model
    logger.info("Training model...")
    model.fit(train[NUMERICS + CATEGORIES], train[OUTPUT])

    # Serialize trained model artifacts
    logger.info("Save model artifacts")
    dump(model, "./data/trained_model")
