import pytest
import pandas as pd
import numpy as np

from production_demo import service
from production_demo.service import InferenceHandler, start_server, main
from production_demo.constants import TRAINED_MODEL_NAME
from sklearn.pipeline import Pipeline
from unittest.mock import MagicMock, call, ANY


def test_inference_model_fn(monkeypatch):
    """Test model function handler
    
    Test that `model_fn` imports a model from some file location.
    """
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
    """Test inference function handler
    
    Test that `predict_fn` launches a sklearn.pipeline.Pipeline predict 
    method using input data.
    """
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
    """Test input parsing function
    
    Test that `input_fn` can read JSON and outputs a pd.DataFrame.
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
    """Test output serialization function
    
    Test that `output_fn` can process predictions data into a CSV string.
    """
    # GIVEN
    prediction = np.array([150000.25838, 200000.5])

    # WHEN
    dih = InferenceHandler()
    x1 = dih.output_fn(prediction)

    # THEN
    assert x1 == "150000.2584\n200000.5000\n"


def test_flask_app(monkeypatch):
    """Test Flask application creation
    
    Test creation of flask application with routes
    """
    # GIVEN
    mock_flask = MagicMock()
    mock_model = MagicMock()
    monkeypatch.setattr(service, "Flask", mock_flask)
    monkeypatch.setattr(InferenceHandler, "model_fn", mock_model)

    # WHEN
    app = start_server()

    # THEN
    assert mock_flask.mock_calls[:2] == [
        call("production_demo.service"),
        call().route("/invocations", methods=["POST"]),
    ]
    assert mock_model.mock_calls == [call("/opt/ml/model")]


def test_request_handler(monkeypatch):
    """Test end to end request handler

    Here we're mocking every step in our InferenceHandler process
    """
    # GIVEN
    input_data_js = b'{"1stFlrSF":896,"2ndFlrSF":0,"BedroomAbvGr":2}\n'
    mock_request = MagicMock()
    mock_model = MagicMock()
    mock_predict = MagicMock()
    mock_input = MagicMock()
    mock_output = MagicMock()
    monkeypatch.setattr(InferenceHandler, "input_fn", mock_input)
    monkeypatch.setattr(InferenceHandler, "predict_fn", mock_predict)
    monkeypatch.setattr(InferenceHandler, "output_fn", mock_output)
    mock_request.configure_mock(
        **{"content_type": "application/json", "data": input_data_js}
    )

    # WHEN
    dih = InferenceHandler()
    x = dih.handle_request(mock_request, mock_model)

    # THEN
    assert mock_input.mock_calls == [
        call(input_data=input_data_js, content_type="application/json"),
    ]
    assert mock_predict.mock_calls == [call(mock_input.return_value, mock_model)]
    assert mock_output.mock_calls == [call(mock_predict.return_value)]
    assert x is not None


def test_main(monkeypatch):
    """Mock our entrypoint
    
    Test launching a web application with Gunicorn
    """
    # GIVEN
    mock_subproc = MagicMock()
    monkeypatch.setattr(service, "Popen", mock_subproc)

    # WHEN
    try:
        main()
    except Exception:
        pass

    # THEN
    assert mock_subproc.mock_calls == [
        call(
            ["gunicorn", "-w", ANY, "-b", ANY, "production_demo.service:start_server()"]
        ),
        call().wait(),
    ]
