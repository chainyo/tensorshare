"""Async http client for sending tensors to a remote server."""

import aiohttp

from tensorshare.client.base import TensorShareClient
from tensorshare.client.utils import fake_tensorshare_data
from tensorshare.schema import TensorShare


class AsyncTensorShareClient(TensorShareClient):
    """Asynchronous Client for sending tensors to a remote server."""

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
        super().__init__(server_url, timeout)

    async def ping_server(self) -> bool:
        """Ping the server to check if it is available."""
        async with aiohttp.ClientSession() as session:
            response = await session.get(str(self.server.ping), timeout=self.timeout)

        return response.status == 200

    async def send_tensor(self, tensor_data: TensorShare) -> aiohttp.ClientResponse:
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

    async def _validate_endpoints(self) -> None:
        """
        Register the endpoints for the client to use.

        Raises:
            ValueError: If the server is not available after the ping request.
        """
        # TODO: Add logging to indicate which endpoint is being checked.
        # Check the ping endpoint
        await self.ping_server()
        # Check the receive_tensor endpoint
        await self.send_tensor(fake_tensorshare_data())


# import torch

# from tensorshare.schema import TensorShare

# ts = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})
# client = AsyncTensorShareClient("http://localhost:8765")
# # client = AsyncTensorShareClient("htt:")

# async def main():
#     r = await client.ping_server()
#     print(r)
#     r = await client.send_tensor(ts)
#     print(r)

# import asyncio
# asyncio.run(main())
