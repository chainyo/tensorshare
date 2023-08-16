"""Test the server router from TensorShare."""

import re
from typing import Dict, List

import pytest
import torch
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
from pydantic import BaseModel

from tensorshare.schema import DefaultResponse, TensorShare
from tensorshare.server import (
    create_async_tensorshare_router,
    create_tensorshare_router,
    default_compute_operation,
)


class TestComputeOperation:
    """Test the compute operation method."""

    def test_default_compute_operation(self) -> None:
        """Test the compute operation method."""
        assert default_compute_operation is not None
        assert callable(default_compute_operation)

        assert default_compute_operation(tensors=1) == DefaultResponse(
            message="Success!"
        )


class TestFastAPIIntegration:
    """Test the FastAPI integration."""

    def test_router_creation_with_default_params(self) -> None:
        """Test the router creation method with default params."""
        router = create_tensorshare_router()

        # Step 1: Check the router
        assert router is not None
        assert isinstance(router, APIRouter)

        assert router.routes is not None
        assert isinstance(router.routes, list)
        assert len(router.routes) == 2
        assert router.routes[0].path == "/receive_tensor"
        assert router.routes[1].path == "/ping"

        # Step 2: Check the FastAPI app with the router
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

    def test_router_creation_with_custom_params(self) -> None:
        """Test the router creation method with custom params."""

        class GoBrrBrr(BaseModel):
            """Mock model for testing."""

            tensor: List[List[float]]

        def compute_go_brr_brr(tensors: TensorShare) -> Dict[str, List[float]]:
            """Mock compute operation for testing."""
            torch_tensors = tensors.to_tensors(backend="torch")

            tensor_to_return = {"tensor": []}
            for _, v in torch_tensors.items():
                item = v.tolist()[0]
                tensor_to_return["tensor"].append(item)

            return tensor_to_return

        server_config = {
            "url": "http://localhost:12345",
            "ping": "healthcheck",
            "receive_tensor": "get_those_tensors",
            "response_model": GoBrrBrr,
        }
        router = create_tensorshare_router(
            server_config=server_config,
            custom_operation=compute_go_brr_brr,
        )

        # Step 1: Check the router
        assert router is not None
        assert isinstance(router, APIRouter)

        assert router.routes is not None
        assert isinstance(router.routes, list)
        assert len(router.routes) == 2
        assert router.routes[0].path == "/get_those_tensors"
        assert router.routes[1].path == "/healthcheck"

        # Step 2: Check the FastAPI app with the router
        app = FastAPI()
        app.include_router(router)

        assert app is not None
        assert isinstance(app, FastAPI)

        client = TestClient(app)
        assert client is not None
        assert isinstance(client, TestClient)

        response = client.get("/healthcheck/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "The TensorShare router is up and running!"
        }

        response = client.post("/get_those_tensors/", data={"shared_tensor": 1})
        assert response.status_code == 422

        ts = TensorShare.from_dict(tensors={"embeddings": torch.ones(1, 1)})
        response = client.post(
            "/get_those_tensors/",
            headers={"Content-Type": "application/json"},
            data=ts.model_dump_json(),
        )
        assert response.status_code == 200
        assert response.json() == {"tensor": [[1.0]]}

    @pytest.mark.asyncio
    async def test_router_async_creation_with_default_params(self) -> None:
        """Test the router creation method with default params."""
        router = create_async_tensorshare_router()

        # Step 1: Check the router
        assert router is not None
        assert isinstance(router, APIRouter)

        assert router.routes is not None
        assert isinstance(router.routes, list)
        assert len(router.routes) == 2
        assert router.routes[0].path == "/receive_tensor"
        assert router.routes[1].path == "/ping"

        # Step 2: Check the FastAPI app with the router
        app = FastAPI()
        app.include_router(router)

        assert app is not None
        assert isinstance(app, FastAPI)

        client = AsyncClient(app=app, base_url="http://testserver")
        assert client is not None
        assert isinstance(client, AsyncClient)

        response = await client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {
            "message": "The TensorShare router is up and running!"
        }

        response = await client.post("/receive_tensor", data={"shared_tensor": 1})
        assert response.status_code == 422

        ts = TensorShare.from_dict(tensors={"embeddings": torch.ones(1, 1)})
        response = await client.post(
            "/receive_tensor",
            headers={"Content-Type": "application/json"},
            data=ts.model_dump_json(),
        )
        assert response.status_code == 200
        assert response.json() == {"message": "Success!"}

    @pytest.mark.asyncio
    async def test_router_async_creation_with_custom_params(self) -> None:
        """Test the router creation method with custom params."""

        class GoBrrBrr(BaseModel):
            """Mock model for testing."""

            tensor: List[List[float]]

        async def compute_go_brr_brr(tensors: TensorShare) -> Dict[str, List[float]]:
            """Mock compute operation for testing."""
            torch_tensors = tensors.to_tensors(backend="torch")

            tensor_to_return = {"tensor": []}
            for _, v in torch_tensors.items():
                item = v.tolist()[0]
                tensor_to_return["tensor"].append(item)

            return tensor_to_return

        server_config = {
            "url": "http://localhost:12345",
            "ping": "healthcheck",
            "receive_tensor": "get_those_tensors",
            "response_model": GoBrrBrr,
        }
        router = create_async_tensorshare_router(
            server_config=server_config,
            custom_operation=compute_go_brr_brr,
        )

        # Step 1: Check the router
        assert router is not None
        assert isinstance(router, APIRouter)

        assert router.routes is not None
        assert isinstance(router.routes, list)
        assert len(router.routes) == 2
        assert router.routes[0].path == "/get_those_tensors"
        assert router.routes[1].path == "/healthcheck"

        # Step 2: Check the FastAPI app with the router
        app = FastAPI()
        app.include_router(router)

        assert app is not None
        assert isinstance(app, FastAPI)

        client = AsyncClient(app=app, base_url="http://testserver")
        assert client is not None
        assert isinstance(client, AsyncClient)

        response = await client.get("/healthcheck")
        assert response.status_code == 200
        assert response.json() == {
            "message": "The TensorShare router is up and running!"
        }

        response = await client.post("/get_those_tensors", data={"shared_tensor": 1})
        assert response.status_code == 422

        ts = TensorShare.from_dict(tensors={"embeddings": torch.ones(1, 1)})
        response = await client.post(
            "/get_those_tensors",
            headers={"Content-Type": "application/json"},
            data=ts.model_dump_json(),
        )
        assert response.status_code == 200
        assert response.json() == {"tensor": [[1.0]]}

    def test_router_creation_with_invalid_function(self) -> None:
        """Test the router creation method with invalid function."""

        async def invalid_function() -> None:
            pass

        with pytest.raises(
            TypeError,
            match=re.escape(
                "The `custom_operation` callable must be synchronous. "
                "Please use the `create_async_tensorshare_router` function instead."
            ),
        ):
            create_tensorshare_router(custom_operation=invalid_function)

    @pytest.mark.asyncio
    async def test_router_async_creation_with_invalid_function(self) -> None:
        """Test the router creation method with invalid function."""

        def invalid_function() -> None:
            pass

        with pytest.raises(
            TypeError,
            match=re.escape(
                "The `custom_operation` callable must be asynchronous. "
                "Please use the `create_tensorshare_router` function instead."
            ),
        ):
            create_async_tensorshare_router(custom_operation=invalid_function)
