""" Launch model hosting server 
"""

from sagemaker_inference import model_server
import production_demo.handler_service

HANDLER_SERVICE = production_demo.handler_service.__file__


def start_server():
    model_server.start_model_server(handler_service=HANDLER_SERVICE)
