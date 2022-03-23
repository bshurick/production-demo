import hashlib
import logging
import pandas as pd

from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from production_demo.constants import (
    CATEGORIES,
    EVAL_SPLITS,
    MODEL_PARAMS,
    NUMERICS,
    OUTPUT,
)
from production_demo.train import CategoriesTransformer


# set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)-15s: %(funcName)s:%(lineno)d: %(message)s",
)

logger = logging.getLogger(__name__)


def handler():
    logger.info("Loading train data")
    train = pd.read_csv("./data/train.csv")

    # log some metrics
    logger.info(train.shape)
    logger.info(train[CATEGORIES + NUMERICS].shape)
    logger.info(train[CATEGORIES + NUMERICS].head().to_markdown())
    logger.info(train[NUMERICS + [OUTPUT]].describe().T.to_markdown())

    # prepare train/test splitting
    train.sort_values(by=["YrSold", "MoSold"], inplace=True)
    tss = TimeSeriesSplit(n_splits=5)

    # set up training pipeline
    logger.info("Running cross validation...")
    hct = CategoriesTransformer(CATEGORIES)
    model = Pipeline(
        [
            ("hash", hct),
            ("LGBM", LGBMRegressor(random_state=22)),
        ]
    )
    model.set_params(**MODEL_PARAMS)
    cv_results = cross_validate(
        estimator=model,
        X=train[NUMERICS + CATEGORIES],
        y=train[OUTPUT],
        cv=tss,
        n_jobs=-1,
        scoring="neg_mean_squared_log_error",
    )

    # Print results to logs
    logger.info("Logging...")
    for k, v in cv_results.items():
        logger.info(f"{k}: {v.mean():.4f}")
