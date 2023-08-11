"""Http client for sending tensors to a remote server."""

from typing import Optional, Union

import requests

from tensorshare.schema import TensorShare


class TensorShareClient:
    """Synchronous Client for sending tensors to a remote server."""

    def __init__(self, server_url: str, timeout: int = 10) -> None:
        """Initialize the client.

        Args:
            server_url (str):
                The url of the server to send tensors to.
            timeout (int):
                The timeout in seconds for the http requests. Defaults to 10.
        """
        self.server_url = server_url
        self.timeout = timeout

    def __enter__(self) -> "Client":
        """Enter the client context."""
        return self

    def __exit__(
        self,
        exception_type: Optional[Union[ValueError, TypeError, AssertionError]],
        exception_value: Optional[Exception],
        traceback: Optional[Exception],
    ) -> None:
        """Exit the client context."""
        pass

    def send_tensor(self, tensor_data: TensorShare) -> None:
        """
        Send a tensor to the remote server.
        """
        response = requests.post(
            f"{self.server_url}/receive_tensor/",
            data=tensor_data.model_dump_json(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response


import torch


tensor_data = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})
with TensorShareClient("http://localhost:8765") as client:
    client.send_tensor(tensor_data)
