import pytest
import pandas as pd
import numpy as np

from production_demo import service
from production_demo.service import InferenceHandler
from production_demo.constants import TRAINED_MODEL_NAME
from sklearn.pipeline import Pipeline
from unittest.mock import MagicMock, call, ANY


def test_inference_model_fn(monkeypatch):
    """Test model function handler"""
    # GIVEN
    mock_load = MagicMock()
    monkeypatch.setattr(service, "load", mock_load)
    mock_load.return_value = Pipeline([("Step", None)])

    # WHEN
    dih = InferenceHandler()
    model = dih.model_fn(".")

    # THEN
    assert mock_load.mock_calls == [call(f"./{TRAINED_MODEL_NAME}")]
    assert model is not None


def test_inference_predict_fn(monkeypatch):
    """Test inference function handler"""
    # GIVEN
    mock_predict = MagicMock()
    mock_data = MagicMock()
    monkeypatch.setattr(Pipeline, "predict", mock_predict)
    monkeypatch.setattr(pd, "DataFrame", mock_data)
    model = Pipeline([("Step", None)])
    data = pd.DataFrame([])
    mock_predict.return_value = ""

    # WHEN
    dih = InferenceHandler()
    x = dih.predict_fn(data, model)

    # THEN
    assert mock_predict.mock_calls == [call(mock_data([]))]
    assert x is not None


def test_inference_input_fn():
    """ Test input parsing function 
    """
    # GIVEN 
    input_data_js = b'{"1stFlrSF":896,"2ndFlrSF":0,"BedroomAbvGr":2}\n'

    # WHEN 
    dih = InferenceHandler()
    x1 = dih.input_fn(input_data_js, "application/json")

    # THEN 
    assert isinstance(x1, pd.DataFrame)
    np.testing.assert_array_equal(x1.values, [[896, 0, 2]])
    assert list(x1.columns) == list(("1stFlrSF", "2ndFlrSF", "BedroomAbvGr"))


def test_inference_output_fn():
    """ Test output serialization function
    """
    # GIVEN 
    prediction = np.array([150000.25838, 200000.5])

    # WHEN 
    dih = InferenceHandler()
    x1 = dih.output_fn(prediction)

    # THEN 
    assert x1 == '150000.2584\n200000.5000\n'
