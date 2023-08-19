"""Async http client for sending tensors to a remote server."""

import asyncio

import aiohttp

from tensorshare.logging import logger
from tensorshare.schema import TensorShare, TensorShareServer
from tensorshare.utils import fake_tensorshare_data


class TensorShareClient:
    """Asynchronous Client for sending tensors to a remote server.

    This client uses the aiohttp library which is an asynchronous http client,
    built on top of asyncio. But the client provides a synchronous interface
    for the users through the use of the `asyncio.run` function.

    Note:
        Using the synchronous interface will block the main thread until the
        request is completed and the response is received. It's not recommended
        to use the synchronous interface in a production environment as it
        may cause performance issues. The synchronous interface is only
        provided for testing purposes.

    Example:
        >>> import torch
        >>> from tensorshare import TensorShare, TensorShareClient, TensorShareServer

        >>> server_config = TensorShareServer("http://localhost:8765")
        >>> client = TensorShareClient(server_config)

        >>> # Synchronous interface
        >>> print(client.ping_server())
        >>> # True

        >>> ts = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})
        >>> print(client.send_tensor(ts))
        >>> # <ClientResponse(http://localhost:8765/receive_tensor) [200 OK]>

        >>> # Asynchronous interface
        >>> import asyncio

        >>> async def main():
        ...     r = await client.async_ping_server()
        ...     print(r)
        ...     r = await client.async_send_tensor(ts)
        ...     print(r)

        >>> asyncio.run(main())
        >>> # True
        >>> # <ClientResponse(http://localhost:8765/receive_tensor) [200 OK]>
    """

    def __init__(
        self,
        server_config: TensorShareServer,
        timeout: int = 10,
        validate_endpoints: bool = True,
    ) -> None:
        """Initialize the client.

        Args:
            server_config (TensorShareServer):
                The server configuration to use for sending tensors.
            timeout (int):
                The timeout in seconds for the http requests. Defaults to 10.
            validate_endpoints (bool):
                Whether to validate the endpoints on the server at the time of
                initialization. Defaults to True.
        """
        self.server = server_config
        self.timeout = timeout

        if validate_endpoints:
            self._validate_endpoints()

    def ping_server(self) -> bool:
        """Ping the server to check if it is available."""
        logger.warning_once(
            "Using the synchronous interface for the client is not recommended."
            " It may cause performance issues, and is only provided for testing."
            " Please use the asynchronous interface instead: `async_ping_server`."
        )
        return asyncio.run(self.async_ping_server())

    async def async_ping_server(self) -> bool:
        """Ping the server to check if it is available."""
        async with aiohttp.ClientSession() as session:
            response = await session.get(str(self.server.ping), timeout=self.timeout)

        _is_available: bool = response.status == 200

        return _is_available

    def send_tensor(self, tensor_data: TensorShare) -> aiohttp.ClientResponse:
        """Send a TensorShare object to the server using aiohttp.

        Args:
            tensor_data (TensorShare):
                The tensor data to send to the server.

        Returns:
            aiohttp.ClientResponse: The response from the server.
        """
        logger.warning_once(
            "Using the synchronous interface for the client is not recommended."
            " It may cause performance issues, and is only provided for testing."
            " Please use the asynchronous interface instead: `async_send_tensor`."
        )
        return asyncio.run(self.async_send_tensor(tensor_data))

    async def async_send_tensor(
        self, tensor_data: TensorShare
    ) -> aiohttp.ClientResponse:
        """
        Send a TensorShare object to the server using aiohttp.

        Args:
            tensor_data (TensorShare):
                The tensor data to send to the server.

        Returns:
            aiohttp.ClientResponse: The response from the server.
        """
        if not isinstance(tensor_data, TensorShare):
            raise TypeError(
                "Expected tensor_data to be of type TensorShare, got"
                f" {type(tensor_data)}"
            )

        async with aiohttp.ClientSession() as session:
            response = await session.post(
                str(self.server.receive_tensor),
                headers={"Content-Type": "application/json"},
                data=tensor_data.model_dump_json(),
                timeout=self.timeout,
            )

        return response

    def _validate_endpoints(self) -> None:
        """
        Check that all the registered endpoints are available on the server.

        Raises:
            AssertionError: If any of the endpoints are not available.
        """
        logger.info("Validating endpoints on the server...")

        try:
            assert asyncio.run(self.async_ping_server()) is True
            logger.info("ping endpoit ✅")

            assert self.send_tensor(fake_tensorshare_data()).status == 200
            logger.info("receive_tensor endpoint ✅")

        except Exception as e:
            raise AssertionError(
                f"Could not connect to the server at {self.server.url}\n{e}"
            ) from e
