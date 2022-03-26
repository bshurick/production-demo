"""Model Hosting Server
"""

from production_demo.service import InferenceHandler
from sagemaker_inference.transformer import Transformer


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """

    def __init__(self):
        transformer = Transformer(default_inference_handler=InferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)


_service = HandlerService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
