"""The Server module introduces all the routing, functions and dependencies for server side."""

import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Type, Union

from fastapi import APIRouter, Depends
from fastapi import status as http_status
from pydantic import BaseModel

from tensorshare.schema import DefaultResponse, TensorShare, TensorShareServer


def default_compute_operation(tensors: TensorShare) -> DefaultResponse:
    """Default action to perform when receiving a TensorShare object."""
    return DefaultResponse(message="Success!")


async def default_async_compute_operation(tensors: TensorShare) -> DefaultResponse:
    """Default asynchronous action to perform when receiving a TensorShare object."""
    return DefaultResponse(message="Success!")


def create_tensorshare_router(
    server_config: Optional[Dict[str, Union[str, Type[BaseModel]]]] = None,
    custom_operation: Optional[Callable] = None,
) -> APIRouter:
    """
    Create and configure a new TensorShare router using the given configuration and computation operation.

    Args:
        server_config (Optional[Dict[str, Union[str, Type[BaseModel]]]]):
            The server configuration to use for the router.
        custom_operation (Optional[Callable]):
            The custom computation operation to use for the `/receive_tensor` endpoint.

    Returns:
        APIRouter:
            The configured TensorShare router ready to be integrated in a FastAPI app.
    """
    if custom_operation is not None and inspect.iscoroutinefunction(custom_operation):
        raise TypeError(
            "The `custom_operation` callable must be synchronous. "
            "Please use the `create_async_tensorshare_router` function instead."
        )

    router = APIRouter()

    _default_config: Dict[str, Union[str, Type[BaseModel]]] = {
        "url": "http://localhost:8000",
        "response_model": DefaultResponse,
    }
    config: TensorShareServer = TensorShareServer.from_dict(
        server_config=server_config or _default_config
    )

    target_operation: Callable = custom_operation or default_compute_operation

    def get_computation_operation() -> Callable:
        """Dependency that returns the currently set computation function."""
        return target_operation

    @router.post(
        f"{config.receive_tensor.path}",
        response_model=config.response_model,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    def receive_tensor(
        shared_tensor: TensorShare,
        operation: Callable = Depends(get_computation_operation),
    ) -> Any:
        """Synchronous endpoint to handle tensors reception and computation."""
        result = operation(shared_tensor)

        return result

    @router.get(
        f"{config.ping.path}",
        response_model=DefaultResponse,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    def ping() -> DefaultResponse:
        """Synchronous endpoint to check if the server is up."""
        return DefaultResponse(message="The TensorShare router is up and running!")

    return router


def create_async_tensorshare_router(
    server_config: Optional[Dict[str, Union[str, Type[BaseModel]]]] = None,
    custom_operation: Optional[Callable[[TensorShare], Awaitable[Any]]] = None,
) -> APIRouter:
    """
    Create and configure a new async TensorShare router using the given configuration and computation operation.

    Args:
        server_config (Optional[Dict[str, Union[str, Type[BaseModel]]]]):
            The server configuration to use for the router.
        custom_operation (Optional[Callable[[TensorShare], Awaitable[Any]]]):
            The async custom computation operation to use for the `/receive_tensor` endpoint.

    Returns:
        APIRouter:
            The configured TensorShare router ready to be integrated in a FastAPI app.
    """
    if custom_operation is not None and not inspect.iscoroutinefunction(
        custom_operation
    ):
        raise TypeError(
            "The `custom_operation` callable must be asynchronous. "
            "Please use the `create_tensorshare_router` function instead."
        )

    router = APIRouter()

    _default_config: Dict[str, Union[str, Type[BaseModel]]] = {
        "url": "http://localhost:8000",
        "response_model": DefaultResponse,
    }
    config: TensorShareServer = TensorShareServer.from_dict(
        server_config=server_config or _default_config
    )

    target_operation: Callable[[TensorShare], Awaitable[Any]] = (
        custom_operation or default_async_compute_operation
    )

    def get_computation_operation() -> Callable[[TensorShare], Awaitable[Any]]:
        """Dependency that returns the currently set async computation function."""
        return target_operation

    @router.post(
        f"{config.receive_tensor.path}",
        response_model=config.response_model,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    async def receive_tensor(
        shared_tensor: TensorShare,
        operation: Callable[[TensorShare], Awaitable[Any]] = Depends(
            get_computation_operation
        ),
    ) -> Any:
        """Asynchronous endpoint to handle tensors reception and computation."""
        result = await operation(shared_tensor)

        return result

    @router.get(
        f"{config.ping.path}",
        response_model=DefaultResponse,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    async def ping() -> DefaultResponse:
        """Asynchronous endpoint to check if the server is up."""
        return DefaultResponse(message="The TensorShare router is up and running!")

    return router
