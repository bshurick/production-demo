import pandas as pd
import pytest

from production_demo import train
from production_demo.train import handler
from production_demo.constants import CATEGORIES, NUMERICS, MODEL_PARAMS, OUTPUT

from unittest.mock import MagicMock, call, ANY


def test_train_handler(monkeypatch):
    # GIVEN
    mock_pandas = MagicMock()
    mock_pipeline = MagicMock()
    mock_dump = MagicMock()
    monkeypatch.setattr(pd, "read_csv", mock_pandas)
    monkeypatch.setattr(train, "Pipeline", mock_pipeline)
    monkeypatch.setattr(train, "dump", mock_dump)

    # WHEN
    handler()

    # THEN
    # Ensure data is loaded and it collects the correct inputs and outputs
    assert mock_pandas.mock_calls == [
        # read CSV
        call(ANY),
        # get X
        call().__getitem__(NUMERICS + CATEGORIES),
        # get y
        call().__getitem__(OUTPUT),
    ]

    # Ensure a pipeline model is created and fitted
    assert mock_pipeline.mock_calls == [
        # create pipeline
        call(ANY),
        # set parameters
        call().set_params(**MODEL_PARAMS),
        # fit x, y
        call().fit(
            mock_pandas().__getitem__(NUMERICS + CATEGORIES),
            mock_pandas().__getitem__(OUTPUT),
        ),
    ]

    # Ensure model artifacts are saved at the end
    assert mock_dump.mock_calls == [
        # dump model artifacts
        call(mock_pipeline(), ANY)
    ]
