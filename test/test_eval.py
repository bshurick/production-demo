import csv
import pandas as pd
import pytest

from io import StringIO
from numpy import array
from production_demo import evaluate, constants
from production_demo.evaluate import handler
from production_demo.constants import (
    CATEGORIES,
    NUMERICS,
    MODEL_PARAMS,
    EVAL_METRIC,
    EVAL_SPLITS,
    OUTPUT,
)

from unittest.mock import MagicMock, call, ANY
from testfixtures import LogCapture


def test_eval_handler(capsys, monkeypatch):
    """Test handler function for evaluation module

    Steps in the process:
        1. **GIVEN** mocked pandas, sklearn scaffolding and cross validation modules...
        2. **WHEN** handler is executed...
        3. **THEN** data is loaded with pandas, the model pipeline is created and cross validated,
           and cross validation results are emitted to logs in the expected format.
    """
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
    handler()
    captured = StringIO(capsys.readouterr().out)

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
            scoring=EVAL_METRIC,
        ),
    ]

    # Examine logs
    reader = csv.reader(captured)
    captured_list = [r for r in reader]
    assert captured_list[0] == [
        "fit_time_mean",
        "fit_time_std",
        "score_time_mean",
        "score_time_std",
        "test_score_mean",
        "test_score_std",
        "test_metric",
        "runtime",
    ]
    assert captured_list[1][:7] == [
        "0.3061",
        "0.1258",
        "0.0347",
        "0.0052",
        "0.0236",
        "0.0046",
        EVAL_METRIC,
    ]
