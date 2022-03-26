
from production_demo import model_server as launchable_model_server
from sagemaker_inference import model_server


def start_server():
    model_server.start_model_server(handler_service=launchable_model_server)
