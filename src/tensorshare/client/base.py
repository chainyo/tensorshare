"""Base Client for sending tensors to a remote server."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, HttpUrl, ValidationError

from tensorshare.schema import TensorShare


class TensorShareServer(BaseModel):
    """ServerUrl model."""

    url: HttpUrl
    ping: HttpUrl
    receive_tensor: HttpUrl


class TensorShareClient(ABC):
    """Base Client for sending tensors to a remote server."""

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

    @abstractmethod
    def ping_server(self) -> bool:
        """Ping the server to check if it is up."""
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, tensor_data: TensorShare) -> None:
        """Send a TensorShare object to the server."""
        raise NotImplementedError

    @abstractmethod
    def _validate_endpoints(self) -> None:
        """Register the endpoints for the server."""
        raise NotImplementedError
