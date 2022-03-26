"""Invoke function

This module contains logic for processing batches of
input data into predictions.

We'll use logic from the SageMaker Inference Toolkit
to create an inference handler class.
https://github.com/aws/sagemaker-inference-toolkit
"""

import os

from flask import Flask, request
from sagemaker_inference import (
    decoder,
    encoder,
)
from subprocess import Popen

from joblib import load
from production_demo.constants import TRAINED_MODEL_NAME


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

        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        return decoder.decode(input_data, content_type)

    def output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        return encoder.encode(prediction, accept)


app = None
def start_server():
    global app
    app = Flask(__name__)
    handler = InferenceHandler()
    model = handler.model_fn('/opt/ml/model')

    @app.route("/invocations", methods=["POST"])
    def invoke(input_data):
        data = handler.input_fn(input_data=input_data,
                                content_type=request.content_type)
        prediction = handler.predict_fn(data, model)
        output = handler.output_fn(prediction, "CSV")
        return output


def main():
    p = Popen(["gunicorn", "-w", "4", "production_demo.service:start_server"]).wait()
    raise Exception(p)
