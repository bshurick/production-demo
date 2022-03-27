"""Invoke function

This module contains logic for processing batches of
input data into predictions.

We'll use logic from the SageMaker Inference Toolkit
to create an inference handler class.
https://github.com/aws/sagemaker-inference-toolkit
"""

import os
import pandas as pd
import logging

from io import StringIO
from flask import Flask, request, Response
from subprocess import Popen
from time import time

from joblib import load
from production_demo.constants import NUMERICS, CATEGORIES, TRAINED_MODEL_NAME

# set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)-15s: %(funcName)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


class InferenceHandler:
    """Default handler for model inference

    This class implements `default_model_fn` and `default_predict_fn`.

    `default_model_fn`: Load a model
    `default_input_fn`: Decode data for inference
    `default_predict_fn`: Make prediction with model
    `default_output_fn`: Serialize data for output
    """

    def model_fn(self, model_dir):
        """Load a Production model

        Load a model using joblib. The model artifacts are already saved locally
        in the Docker environment by SageMaker.

        :param model_dir: The directory where a model is saved (e.g. /opt/ml)
        :type model_dir: str
        :returns: A Sklearn Pipeline object
        :rtype: sklearn.pipeline.Pipeline
        """
        model = load(os.path.join(model_dir, TRAINED_MODEL_NAME))
        return model

    def predict_fn(self, data, model):
        """Make prediction with model

        Predicting with an Sklearn model is straightforward.

        :param data: Input data (pandas.Dataframe) for prediction
        :type data: pandas.DataFrame

        :param model: An Sklearn Pipeline model
        :type model: sklearn.pipeline.Pipeline
        :returns: A pandas dataframe to be deserialized
        :rtype: pandas.DataFrame
        """
        return model.predict(data)

    def input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized pandas object
        """
        assert content_type in ["text/csv", "application/json"]
        f = StringIO()
        f.write(input_data)
        f.seek(0)
        if content_type == "text/csv":
            input_df = pd.read_csv(f, names=NUMERICS + CATEGORIES)
        else:
            input_df = pd.read_json(f, lines=True)
        return input_df

    def output_fn(self, prediction):
        """Serializes predictions from predict_fn to CSV

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        f = StringIO()
        prediction.to_csv(f, index=False, header=False)
        f.seek(0)
        return f.getvalue()


def start_server():
    app = Flask(__name__)
    handler = InferenceHandler()
    model = handler.model_fn("/opt/ml/model")

    @app.route("/invocations", methods=["POST"])
    def invoke():
        start = time()
        data = handler.input_fn(
            input_data=request.data, 
            content_type=request.content_type
        )
        prediction = handler.predict_fn(data, model)
        output = handler.output_fn(prediction)
        end = time()
        logger.info(f"response_time: {end-start:.0f}")
        return Response(output, status=200, mimetype="text/csv")
    
    return app


def main():
    p = Popen(
        [
            "gunicorn",
            "-w",
            "4",
            "-b",
            "127.0.0.1:8000",
            "production_demo.service:start_server",
        ]
    ).wait()
    # register quit as exception
    raise Exception(p)
