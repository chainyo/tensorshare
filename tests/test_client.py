"""Test the Client module."""

import re

import aiohttp
import pytest
import torch
from pydantic import HttpUrl

from tensorshare.schema import TensorShare, TensorShareServer
from tensorshare.client import TensorShareClient


class TestTensorShareClient:
    """Test the TensorShareClient class."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8765",
            "https://localhost:8765",
            "http://abc1234.com",
            "https://abc1234.com",
        ],
    )
    def test_client_init_no_validation(self, url: str) -> None:
        """Test the client init without endpoints validation."""
        client = TensorShareClient(server_url=url, validate_endpoints=False)

        assert isinstance(client.server, TensorShareServer)

        assert client.timeout == 10
        assert isinstance(client.timeout, int)

        assert client.server.url == HttpUrl(url)
        assert str(client.server.url) == f"{url}/"

        assert client.server.ping == HttpUrl(f"{url}/ping")
        assert str(client.server.ping) == f"{url}/ping"

        assert client.server.receive_tensor == HttpUrl(f"{url}/receive_tensor")
        assert str(client.server.receive_tensor) == f"{url}/receive_tensor"

        assert client.server.model_dump() == {
            "url": HttpUrl(url),
            "ping": HttpUrl(f"{url}/ping"),
            "receive_tensor": HttpUrl(f"{url}/receive_tensor"),
        }

        assert hasattr(client, "send_tensor")
        assert callable(client.send_tensor)

        assert hasattr(client, "ping_server")
        assert callable(client.ping_server)

        assert hasattr(client, "async_send_tensor")
        assert callable(client.async_send_tensor)

        assert hasattr(client, "async_ping_server")
        assert callable(client.async_ping_server)

        assert hasattr(client, "_validate_endpoints")
        assert callable(client._validate_endpoints)

    @pytest.mark.parametrize(
        "url",
        [
            "localhost:8765",
            "abc1234.com",
            "ftp://localhost:8765",
            "ftp://abc1234.com",
            "data://localhost:8765",
            "data://abc1234.com",
        ],
    )
    def test_client_with_invalid_url(self, url: str) -> None:
        """Test the client init with invalid url."""
        with pytest.raises(
            ValueError,
            match=re.escape(f"The provided server url is invalid -> {url}\n"),
        ):
            TensorShareClient(server_url=url)

    @pytest.mark.usefixtures("mock_server")
    @pytest.mark.parametrize(
        ["ping_status", "receive_tensor_status"],
        [
            (200, 500),
            (500, 200),
            (500, 500),
        ],
    )
    def test_client_init_with_validation(
        self,
        ping_status: int,
        receive_tensor_status: int,
        mock_server,
    ) -> None:
        """Test the client init with endpoints validation."""
        url = "http://localhost:8765/"

        with pytest.raises(
            AssertionError,
            match=re.escape(f"Could not connect to the server at {url}\n"),
        ):
            mock_server.get(f"{url}/ping", status=ping_status)
            mock_server.post(f"{url}/receive_tensor", status=receive_tensor_status)
            TensorShareClient(server_url=url)

    @pytest.mark.usefixtures("mock_server")
    def test_client_ping_server(self, mock_server) -> None:
        """Test the client ping server."""
        client = TensorShareClient(
            server_url="http://localhost:8765", validate_endpoints=False
        )

        mock_server.get("http://localhost:8765/ping", status=200)
        r = client.ping_server()
        assert r is True

        mock_server.get("http://localhost:8765/ping", status=500)
        r = client.ping_server()
        assert r is False

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_server")
    async def test_client_async_ping_server(self, mock_server) -> None:
        """Test the client async ping server."""
        client = TensorShareClient(
            server_url="http://localhost:8765", validate_endpoints=False
        )

        mock_server.get("http://localhost:8765/ping", status=200)
        r = await client.async_ping_server()
        assert r is True

        mock_server.get("http://localhost:8765/ping", status=500)
        r = await client.async_ping_server()
        assert r is False

    @pytest.mark.usefixtures("mock_server")
    def test_client_send_tensor(self, mock_server) -> None:
        """Test the client send tensor."""
        ts = TensorShare.from_dict({"embeddings": torch.ones(1, 1)})
        client = TensorShareClient(
            server_url="http://localhost:8765", validate_endpoints=False
        )

        mock_server.post("http://localhost:8765/receive_tensor", status=200)
        r = client.send_tensor(tensor_data=ts)
        assert r.status == 200
        assert isinstance(r, aiohttp.ClientResponse)

        mock_server.post("http://localhost:8765/receive_tensor", status=500)
        r = client.send_tensor(tensor_data=ts)
        assert r.status == 500
        assert isinstance(r, aiohttp.ClientResponse)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_server")
    async def test_client_async_send_tensor(self, mock_server) -> None:
        """Test the client async send tensor."""
        ts = TensorShare.from_dict({"embeddings": torch.ones(1, 1)})
        client = TensorShareClient(
            server_url="http://localhost:8765", validate_endpoints=False
        )

        mock_server.post("http://localhost:8765/receive_tensor", status=200)
        r = await client.async_send_tensor(tensor_data=ts)
        assert r.status == 200
        assert isinstance(r, aiohttp.ClientResponse)

        mock_server.post("http://localhost:8765/receive_tensor", status=500)
        r = await client.async_send_tensor(tensor_data=ts)
        assert r.status == 500
        assert isinstance(r, aiohttp.ClientResponse)

    @pytest.mark.parametrize(
        "tensors",
        [
            100,
            [1, 2, 3],
            [torch.ones(1, 1), torch.ones(1, 1)],
            [torch.ones(1, 1), 2],
            {"embeddings": torch.ones(1, 1)},
            {"embeddings": 2},
            "tensors_are_invalid",
            (1, 2, 3),
        ],
    )
    def test_client_send_tensor_with_invalid_tensors(self, tensors) -> None:
        """Test client send_tensor with invalid tensors."""
        client = TensorShareClient(
            server_url="http://localhost:8765", validate_endpoints=False
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Expected tensor_data to be of type TensorShare, got {type(tensors)}"
            ),
        ):
            client.send_tensor(tensor_data=tensors)
