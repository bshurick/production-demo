import pandas as pd
import pytest

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

    # WHEN
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
