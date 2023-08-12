"""Async http client for sending tensors to a remote server."""

import asyncio

import aiohttp
from pydantic import BaseModel, HttpUrl, ValidationError

from tensorshare.client.utils import fake_tensorshare_data
from tensorshare.schema import TensorShare


class TensorShareServer(BaseModel):
    """ServerUrl model."""

    url: HttpUrl
    ping: HttpUrl
    receive_tensor: HttpUrl


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
        >>> from tensorshare import TensorShare, TensorShareClient

        >>> client = TensorShareClient("http://localhost:8765")
        >>> ts = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})

        >>> # Synchronous interface
        >>> print(client.ping_server())
        >>> # True

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

    def __init__(self, server_url: str, timeout: int = 10) -> None:
        """Initialize the client.

        Args:
            server_url (str):
                The url of the server to send tensors to.
            timeout (int):
                The timeout in seconds for the http requests. Defaults to 10.

        Raises:
            ValueError: If the server_url is not a valid url.
        """
        try:
            self.server = TensorShareServer(
                url=HttpUrl(server_url),
                ping=HttpUrl(f"{server_url}/ping"),
                receive_tensor=HttpUrl(f"{server_url}/receive_tensor"),
            )
        except ValidationError as e:
            raise ValueError(
                f"The provided server url is invalid -> {server_url}\n{e}"
            ) from e

        self.timeout = timeout

        self._validate_endpoints()

    def ping_server(self) -> bool:
        """Ping the server to check if it is available."""
        return asyncio.run(self.async_ping_server())

    async def async_ping_server(self) -> bool:
        """Ping the server to check if it is available."""
        async with aiohttp.ClientSession() as session:
            response = await session.get(str(self.server.ping), timeout=self.timeout)

        return response.status == 200

    def send_tensor(self, tensor_data: TensorShare) -> aiohttp.ClientResponse:
        """Send a TensorShare object to the server using aiohttp.

        Args:
            tensor_data (TensorShare):
                The tensor data to send to the server.

        Returns:
            aiohttp.ClientResponse: The response from the server.
        """
        return asyncio.run(self.async_send_tensor(tensor_data))

    async def async_send_tensor(self, tensor_data: TensorShare) -> aiohttp.ClientResponse:
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
            session.close()

        return response

    def _validate_endpoints(self) -> None:
        """
        Check that all the registered endpoints are available on the server.

        Raises:
            AssertionError: If any of the endpoints are not available.
        """
        # TODO: Add logging to indicate which endpoint is being checked.
        try:
            # Check the ping endpoint
            r = self.ping_server()
            assert r is True

            # Check the receive_tensor endpoint
            r = self.send_tensor(fake_tensorshare_data())
            assert r.status == 200

        except Exception as e:
            raise AssertionError(
                f"Could not connect to the server at {self.server.url}\n{e}"
            ) from e


# import torch

# from tensorshare.schema import TensorShare

# ts = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})
# client = TensorShareClient("http://localhost:8765")
# # client = AsyncTensorShareClient("htt:")

# async def main():
#     r = await client.async_ping_server()
#     print(r)
#     r = await client.async_send_tensor(ts)
#     print(r)

# import asyncio
# asyncio.run(main())

# print(client.ping_server())
# print(client.send_tensor(ts))
