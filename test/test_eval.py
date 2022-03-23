import pandas as pd
import pytest

from production_demo import evaluate
from production_demo.evaluate import handler
from production_demo.constants import CATEGORIES, NUMERICS, MODEL_PARAMS, OUTPUT

from unittest.mock import MagicMock, call, ANY


def test_train_handler(monkeypatch):
    # GIVEN
    mock_pandas = MagicMock()
    mock_pipeline = MagicMock()
    mock_cv = MagicMock()
    monkeypatch.setattr(pd, "read_csv", mock_pandas)
    monkeypatch.setattr(evaluate, "Pipeline", mock_pipeline)
    monkeypatch.setattr(evaluate, "cross_validate", mock_cv)

    # WHEN
    handler()

    # THEN
    assert mock_pipeline.mock_calls == [
        # create pipeline
        call(ANY),
        # set parameters
        call().set_params(**MODEL_PARAMS),
    ]
