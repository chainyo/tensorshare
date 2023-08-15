"""The Server module introduces all the routing, functions and dependencies for server side."""

from typing import Any, Callable, Dict, Type, Union

from fastapi import APIRouter, Depends
from fastapi import status as http_status
from pydantic import BaseModel

from tensorshare.schema import DefaultResponse, TensorShare, TensorShareServer

router = APIRouter()

_default_config: Dict[str, Union[str, Type[BaseModel]]] = {
    "url": "http://localhost:8000",
    "response_model": DefaultResponse,
}
server_config: TensorShareServer = TensorShareServer.from_dict(
    server_config=_default_config
)


def compute_operation(tensors: TensorShare) -> DefaultResponse:
    """
    Default action to perform when receiving a TensorShare object with a POST request.

    Args:
        tensors (TensorShare):
            The TensorShare object received from the client.

    Returns:
        DefaultResponse:
            The response model to send back to the client with the result of the computation.
    """
    return DefaultResponse(message="Success!")


target_operation: Callable = compute_operation


def get_computation_operation() -> Callable:
    """Dependency that returns the currently set computation function."""
    return target_operation


@router.post(
    f"{server_config.receive_tensor.path}",
    response_model=server_config.response_model,
    status_code=http_status.HTTP_200_OK,
    tags=["tensorshare"],
)
def receive_tensor(
    shared_tensor: TensorShare, operation: Callable = Depends(get_computation_operation)
) -> Any:
    """Endpoint to handle tensors reception and computation."""
    result = operation(shared_tensor)

    return result


@router.get(
    f"{server_config.ping.path}",
    response_model=DefaultResponse,
    status_code=http_status.HTTP_200_OK,
    tags=["tensorshare"],
)
def ping() -> DefaultResponse:
    """Endpoint to check if the server is up."""
    return DefaultResponse(message="The TensorShare router is up and running!")
