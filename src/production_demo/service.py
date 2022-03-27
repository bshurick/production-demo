"""Flask invocation service

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

    This class provides core functions to load model artifactgs,
    deserialize input data, predict using a trained model, and 
    output a response.
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
        """A default input_fn that can handle JSON format.

        :param input_data: The request payload serialized in the content_type format
        :type input_data: pandas.Dataframe

        :param content_type: The request content_type
        :type content_type: str

        :returns: input_data deserialized pandas object
        :rtype: pandas.DataFrame
        """
        assert content_type in ["application/json"]
        f = StringIO()
        f.write(input_data.decode())
        f.seek(0)
        input_df = pd.read_json(f, lines=True)
        return input_df

    def output_fn(self, prediction):
        """Serializes predictions from predict_fn to CSV

        :param prediction: A prediction result from predict_fn
        :type prediction: numpy.array

        :returns: Output data serialized
        :rtype: str
        """
        f = StringIO()
        for pred in prediction:
            f.write(f'{pred:.4f}\n')
        return f.getvalue()

    def handle_request(self, request, model):
        """Handle a single request
        
        Routes to input, prediction, output methods

        :param request: The request object
        :type request: requests.request

        :param model: The trained model 
        :type model: sklearn.pipeline.Pipeline

        :returns: Output from output_fn
        :rtype: str
        """
        data = self.input_fn(
            input_data=request.data, 
            content_type=request.content_type
        )
        prediction = self.predict_fn(data, model)
        output = self.output_fn(prediction)
        return output


def start_server():
    """Create Flask application 
    """
    app = Flask(__name__)
    handler = InferenceHandler()
    model = handler.model_fn("/opt/ml/model")

    @app.route("/invocations", methods=["POST"])
    def invoke():
        """Model service invocation route
        """
        start = time()
        output = handler.handle_request(request, model)
        end = time()
        logger.info(f"response_time: {end-start:.4f}")
        return Response(output, status=200, mimetype="text/csv")
    
    return app


def main():
    """Entrypoint for launching Flask service
    """
    p = Popen(
        [
            "gunicorn",
            "-w",
            "4",
            "-b",
            "0.0.0.0:8000",
            "production_demo.service:start_server()",
        ]
    ).wait()
    # register quit as exception
    raise Exception(p)
