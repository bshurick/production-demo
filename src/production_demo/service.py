"""Invoke function

This module contains logic for processing batches of
input data into predictions.

We'll use logic from the SageMaker Inference Toolkit
to create an inference handler class.
https://github.com/aws/sagemaker-inference-toolkit
"""

import os

from sagemaker_inference import model_server
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
)
from joblib import load

from production_demo import model_server as launchable_model_server
from production_demo.constants import TRAINED_MODEL_NAME


class InferenceHandler(default_inference_handler.DefaultInferenceHandler):
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


def start_server():
    model_server.start_model_server(handler_service=launchable_model_server)
