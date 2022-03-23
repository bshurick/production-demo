import logging
import pandas as pd
import pytest

from numpy import array
from production_demo import evaluate
from production_demo.evaluate import handler
from production_demo.constants import (
    CATEGORIES,
    NUMERICS,
    MODEL_PARAMS,
    EVAL_SPLITS,
    OUTPUT,
)

from unittest.mock import MagicMock, call, ANY
from testfixtures import LogCapture


def test_eval_handler(monkeypatch):
    # GIVEN
    mock_pandas = MagicMock()
    mock_pipeline = MagicMock()
    mock_cv = MagicMock()
    mock_tss = MagicMock()
    monkeypatch.setattr(pd, "read_csv", mock_pandas)
    monkeypatch.setattr(evaluate, "Pipeline", mock_pipeline)
    monkeypatch.setattr(evaluate, "cross_validate", mock_cv)
    monkeypatch.setattr(evaluate, "TimeSeriesSplit", mock_tss)
    mock_cv.return_value = {
        "fit_time": array([0.13091469, 0.21366191, 0.30524421, 0.39636087, 0.48411179]),
        "score_time": array(
            [0.02884722, 0.03217411, 0.03273487, 0.03577304, 0.04408193]
        ),
        "test_score": array(
            [-0.02741252, -0.02991624, -0.01693261, -0.02269257, -0.0208453]
        ),
    }

    # WHEN
    with LogCapture() as l:
        handler()

    # THEN
    # Ensure a model pipeline is created and model params are set
    assert mock_pipeline.mock_calls == [
        # create pipeline
        call(ANY),
        # set parameters
        call().set_params(**MODEL_PARAMS),
    ]

    # Cross validation is called
    assert mock_cv.mock_calls[:1] == [
        call(
            estimator=mock_pipeline(),
            X=mock_pandas().__getitem__(NUMERICS + CATEGORIES),
            y=mock_pandas().__getitem__(OUTPUT),
            cv=mock_tss(n_splits=EVAL_SPLITS),
            n_jobs=-1,
            scoring="neg_mean_squared_log_error",
        ),
    ]

    # Examine logs
    l.check_present(
        ("production_demo.evaluate", "INFO", "fit_time: 0.3061"),
        ("production_demo.evaluate", "INFO", "score_time: 0.0347"),
        ("production_demo.evaluate", "INFO", "test_score: -0.0236"),
    )
