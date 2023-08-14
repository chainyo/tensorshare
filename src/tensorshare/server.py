"""The Server module introduces all the routing, functions and dependencies for server side."""

from typing import Callable

from fastapi import APIRouter, Depends
from fastapi import status as http_status

from tensorshare.schema import DefaultResponse, TensorShare, TensorShareServer

router = APIRouter()

_default_config = {"url": "http://localhost:8000", "response_model": DefaultResponse}
server_config = TensorShareServer.from_dict(server_config=_default_config)


def compute_operation(tensors: TensorShare) -> server_config.response_model:
    """
    Default action to perform when receiving a TensorShare object with a POST request.

    Args:
        tensors (TensorShare):
            The TensorShare object received from the client.

    Returns:
        server_config.response_model:
            The response model to send back to the client configured in the server_config.
            By default, it is a DefaultResponse object.
    """
    return {"message": "Success"}


@router.post(
    f"{server_config.receive_tensor.path}",
    response_model=server_config.response_model,
    status_code=http_status.HTTP_200_OK,
)
def receive_tensor(
    shared_tensor: TensorShare, operation: Callable = Depends(compute_operation)
) -> server_config.response_model:
    """Endpoint to handle tensors reception and computation."""
    result = operation(shared_tensor)

    return result


@router.get(
    f"{server_config.ping.path}",
    response_model=DefaultResponse,
    status_code=http_status.HTTP_200_OK,
)
def ping() -> DefaultResponse:
    """Endpoint to check if the server is up."""
    return {"message": "The TensorShare router is up and running!"}
