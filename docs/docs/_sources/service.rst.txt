Service Module
==========================================

The service module exposes a Flask application 
with an `/invocations` endpoint that accepts 
POST requests with input features, and responds
with a prediction (or set of predictions). 

.. autosummary::
    :toctree: modules
    :recursive:

    production_demo.service
    production_demo.service.InferenceHandler
    production_demo.service.start_server
    production_demo.service.main
