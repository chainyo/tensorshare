"""Test the server router from TensorShare."""

import torch
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import HttpUrl

try:
    from tensorshare.server import router
except ImportError as e:
    raise ImportError("No `router` found in tensorshare.server") from e

try:
    from tensorshare.server import _default_config
except ImportError as e:
    raise ImportError("No `_default_config` found in tensorshare.server") from e

try:
    from tensorshare.server import server_config
except ImportError as e:
    raise ImportError("No `server_config` found in tensorshare.server") from e

try:
    from tensorshare.server import target_operation
except ImportError as e:
    raise ImportError("No `target_operation` found in tensorshare.server") from e

from tensorshare.schema import DefaultResponse, TensorShare, TensorShareServer
from tensorshare.server import (
    compute_operation,
    get_computation_operation,
    ping,
    receive_tensor,
)


class TestServerVariables:
    """Test the server file variables."""

    def test_router(self) -> None:
        """Test the router."""
        assert router is not None
        assert isinstance(router, APIRouter)

    def test_default_config(self) -> None:
        """Test the default config."""
        assert _default_config is not None
        assert isinstance(_default_config, dict)

        assert "url" in _default_config
        assert _default_config["url"] == "http://localhost:8000"
        assert "response_model" in _default_config
        assert _default_config["response_model"] == DefaultResponse

    def test_server_config(self) -> None:
        """Test the server config."""
        assert server_config is not None
        assert isinstance(server_config, TensorShareServer)

        assert server_config.url == HttpUrl("http://localhost:8000")
        assert server_config.ping == HttpUrl("http://localhost:8000/ping")
        assert server_config.receive_tensor == HttpUrl(
            "http://localhost:8000/receive_tensor"
        )
        assert server_config.response_model == DefaultResponse

    def test_target_operation(self) -> None:
        """Test the target operation."""
        assert target_operation is not None
        assert callable(target_operation)

        assert target_operation == compute_operation


class TestComputeOperation:
    """Test the compute operation method."""

    def test_compute_operation(self) -> None:
        """Test the compute operation method."""
        assert compute_operation is not None
        assert callable(compute_operation)

        assert compute_operation(tensors=1) == DefaultResponse(message="Success!")


class TestGetComputationOperation:
    """Test the get computation operation method."""

    def test_get_computation_operation(self) -> None:
        """Test the get computation operation method."""
        assert get_computation_operation is not None
        assert callable(get_computation_operation)

        assert get_computation_operation() == compute_operation


class TestReceiveTensor:
    """Test the receive tensor method."""

    def test_receive_tensor(self) -> None:
        """Test the receive tensor method."""
        assert receive_tensor is not None
        assert callable(receive_tensor)

        assert receive_tensor(
            shared_tensor=1,
            operation=compute_operation,
        ) == DefaultResponse(message="Success!")


class TestPing:
    """Test the ping method."""

    def test_ping(self) -> None:
        """Test the ping method."""
        assert ping is not None
        assert callable(ping)

        assert ping() == DefaultResponse(
            message="The TensorShare router is up and running!"
        )


class TestFastAPIIntegration:
    """Test the FastAPI integration."""

    def test_fastapi_integration(self) -> None:
        """Test the FastAPI integration."""
        app = FastAPI()
        app.include_router(router)

        assert app is not None
        assert isinstance(app, FastAPI)

        client = TestClient(app)
        assert client is not None
        assert isinstance(client, TestClient)

        response = client.get("/ping/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "The TensorShare router is up and running!"
        }

        response = client.post("/receive_tensor/", data={"shared_tensor": 1})
        assert response.status_code == 422

        ts = TensorShare.from_dict(tensors={"embeddings": torch.ones(1, 1)})
        response = client.post(
            "/receive_tensor/",
            headers={"Content-Type": "application/json"},
            data=ts.model_dump_json(),
        )
        assert response.status_code == 200
        assert response.json() == {"message": "Success!"}
