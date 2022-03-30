"""Training module

This module trains contains a command-line entrypoint to
train and save model from a data/train.csv file.
"""

import hashlib
import logging
import os
import pandas as pd

from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from production_demo.constants import (
    CATEGORIES,
    MODEL_PARAMS,
    NUMERICS,
    OUTPUT,
    TRAINED_MODEL_NAME,
)


# set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)-15s: %(funcName)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


class CategoriesTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for categories

    Use this to create hashed numeric values for a list of
    categorical columns for any pandas dataframe. This can be used
    in place of One Hot Encoding of categorical values, and does not
    require storing a lookup of possible categories for each column.

    The LightGBM package performs best without transforming into a high dimensional
    (i.e. one hot encoded) feature space; instead, we will directly train using
    hashed numeric buckets.
    """

    def __init__(self, category_cols: list, n_buckets: int = 100000):
        """Choose columns to hash into N buckets

        :param category_cols: A list of categorical columns to hash
        :type category_cols: list

        :param n_buckets: The number of hash buckets to use when hashing
        :type n_buckets: int
        """
        self.category_cols = category_cols
        self.n_buckets = n_buckets

    @staticmethod
    def hash_col(x: str, n_buckets: int = 100000) -> int:
        """Hash a string value into numeric buckets

        :param x: A string to hash
        :type x: str

        :param n_buckets: The number of hash buckets to use
        :type n_buckets: int

        :return: The integer bucket value
        :rtype: int
        """
        return int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % n_buckets

    def fit(self, X, y=None):
        """Implements a fit function but does nothing"""
        # no fit; return self
        return self

    def transform(self, X):
        """Apply our hashing function to categorical columns

        Loop through categorical column and replace with numeric hash buckets.
        """
        _X = X.copy()
        for col in self.category_cols:
            _X[col].fillna("", inplace=True)
            _X[col] = _X[col].apply(lambda x: self.hash_col(x, self.n_buckets))
        return _X


def handler():
    """Main entry point for model training

    Steps in the training process:
        1. Load data from ./data/train.csv
        2. Create training pipeline using Sklearn Pipeline
        3. Set model params learned during parameter tuning
        4. Fit model on training data
        5. Serialize model using joblib
    """

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
    dump(model, os.path.join("./data", TRAINED_MODEL_NAME))
