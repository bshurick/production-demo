"""Evaluation / cross-validation module

This module trains contains a command-line entrypoint to
run cross validation using `EVAL_SPLITS` and `MODEL_PARAMS`
from `production_demo.constants` and emit the cross validation
results to logs.
"""

import json
import logging
import numpy as np
import pandas as pd
import sys

from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from production_demo.constants import (
    CATEGORIES,
    EVAL_METRIC,
    EVAL_SPLITS,
    MODEL_PARAMS,
    NUMERICS,
    OUTPUT,
)
from production_demo.train import CategoriesTransformer


# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)-15s: %(funcName)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


def handler():
    """Main entry point for model evaluate script

    Steps in the evaluation process:
        1. Load data from ./data/train.csv
        2. Sort data by date sold and create time-based cross validation splitter
        3. Create model training pipeline using Sklearn
        4. Set model params learned during parameter tuning
        5. Run cross validation
        6. Emit cross validation results into logs
    """

    logger.info("Loading train data")
    train = pd.read_csv("./data/train.csv")

    # Log some metrics
    logger.info(train.shape)
    logger.info(train[NUMERICS + CATEGORIES].shape)
    logger.info(train[NUMERICS + CATEGORIES].head().to_markdown())
    logger.info(train[NUMERICS + [OUTPUT]].describe().T.to_markdown())

    # Prepare train/test splitting
    train.sort_values(by=["YrSold", "MoSold"], inplace=True)
    tss = TimeSeriesSplit(n_splits=EVAL_SPLITS)

    # Set up training pipeline
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
        scoring=EVAL_METRIC,
    )

    # reformat results for output
    _output_results = dict(**cv_results)
    _output_results["test_score"] *= -1
    for k, v in cv_results.items():
        if isinstance(v, np.ndarray):
            _output_results[f"{k}_mean"] = f"{v.mean():.4f}"
            _output_results[f"{k}_std"] = f"{v.std():.4f}"
            _output_results.pop(k)

    _output_results["test_metric"] = EVAL_METRIC
    _output_results["runtime"] = str(datetime.now())

    # Print results to stdout
    logger.info("Logging...")
    log_df = pd.DataFrame([_output_results], index=[0]).to_csv(index=False)
    sys.stdout.write(f"{log_df}\n")
