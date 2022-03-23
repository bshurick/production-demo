import pandas as pd
import pytest

from sklearn.model_selection import TimeSeriesSplit

from production_demo import evaluate
from production_demo.evaluate import handler
from production_demo.constants import MODEL_PARAMS, EVAL_SPLITS

from unittest.mock import MagicMock, call, ANY


def test_eval_handler(monkeypatch):
    # GIVEN
    mock_pandas = MagicMock()
    mock_pipeline = MagicMock()
    mock_cv = MagicMock()
    monkeypatch.setattr(pd, "read_csv", mock_pandas)
    monkeypatch.setattr(evaluate, "Pipeline", mock_pipeline)
    monkeypatch.setattr(evaluate, "cross_validate", mock_cv)
    tss = TimeSeriesSplit(n_splits=EVAL_SPLITS)

    # WHEN
    handler()

    # THEN

    # ensure we create pipeline and set model params
    assert mock_pipeline.mock_calls == [
        # create pipeline
        call(ANY),
        # set parameters
        call().set_params(**MODEL_PARAMS),
    ]

    # test call to cross validation
    assert mock_cv.mock_calls[:1] == [
        call(
            estimator=mock_pipeline(),
            X=ANY,
            y=ANY,
            cv=tss,
            n_jobs=-1,
            scoring="neg_mean_squared_log_error",
        ),
    ]
