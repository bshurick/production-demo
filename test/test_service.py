import pytest
import pandas as pd


from production_demo import service
from production_demo.service import DefaultInferenceHandler
from production_demo.constants import TRAINED_MODEL_NAME
from sklearn.pipeline import Pipeline
from unittest.mock import MagicMock, call, ANY


def test_inference_model_fn(monkeypatch):
    """ Test model function handler
    """
    # GIVEN
    mock_load = MagicMock()
    monkeypatch.setattr(service, "load", mock_load)
    mock_load.return_value = Pipeline([('Step', None)])

    # WHEN
    dih = DefaultInferenceHandler()
    model = dih.default_model_fn('.')

    # THEN
    assert mock_load.mock_calls == [call(f'./{TRAINED_MODEL_NAME}')]
    assert model is not None


def test_inference_predict_fn(monkeypatch):
    """ Test inference function handler
    """
    # GIVEN
    mock_predict = MagicMock()
    mock_data = MagicMock()
    monkeypatch.setattr(Pipeline, "predict", mock_predict)
    monkeypatch.setattr(pd, "DataFrame", mock_data)
    model = Pipeline([('Step', None)])
    data = pd.DataFrame([])
    mock_predict.return_value = ''

    # WHEN
    dih = DefaultInferenceHandler()
    x = dih.default_predict_fn(data, model)

    # THEN
    assert mock_predict.mock_calls == [call(mock_data([]))]
    assert x is not None
