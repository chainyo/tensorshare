"""Tests for the pydantic schemas of tensorshare."""

import base64
import re

import jax.numpy as jnp
import numpy as np
import paddle
import pkg_resources
import pytest

# import tensorflow as tf
import torch
from pydantic import BaseModel, ByteSize, HttpUrl

from tensorshare.schema import DefaultResponse, TensorShare, TensorShareServer
from tensorshare.serialization.constants import Backend
from tensorshare.serialization.flax import serialize as serialize_flax
from tensorshare.serialization.numpy import serialize as serialize_numpy
from tensorshare.serialization.paddle import serialize as serialize_paddle
from tensorshare.serialization.torch import serialize as serialize_torch

# from tensorshare.serialization.tensorflow import serialize as serialize_tensorflow


class TestTensorShare:
    """Tests for the pydantic schemas of tensorshare."""

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_tensorshare_pydantic_schema_with_fixed_tensor(
        self, serialized_fixed_numpy_tensors
    ) -> None:
        """Test the pydantic schema of tensorshare."""
        tensorshare = TensorShare(
            tensors=serialized_fixed_numpy_tensors,
            size=len(serialized_fixed_numpy_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert isinstance(tensorshare.size_as_str, str)
        assert tensorshare.size_as_str == "112B"

    @pytest.mark.usefixtures("random_fp32_numpy_tensors")
    def test_tensorshare_pydantic_schema_with_random_tensor(
        self, random_fp32_numpy_tensors
    ) -> None:
        """Test the pydantic schema of tensorshare."""
        converted_tensors = serialize_numpy(random_fp32_numpy_tensors)
        tensorshare = TensorShare(
            tensors=converted_tensors,
            size=len(converted_tensors),
        )

        assert tensorshare.tensors == converted_tensors
        assert tensorshare.size == len(converted_tensors)

        assert (
            tensorshare.size_as_str == ByteSize(len(converted_tensors)).human_readable()
        )

    @pytest.mark.usefixtures("dict_zeros_flax_tensor")
    def test_tensorshare_with_flax(self, dict_zeros_flax_tensor) -> None:
        """Test the tensorshare schema with flax tensors."""
        _tensors = serialize_flax(dict_zeros_flax_tensor)
        tensorshare = TensorShare(
            tensors=base64.b64encode(_tensors),
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(_tensors)
        assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_numpy_tensor")
    def test_tensorshare_with_numpy(self, dict_zeros_numpy_tensor) -> None:
        """Test the tensorshare schema with numpy tensors."""
        _tensors = serialize_numpy(dict_zeros_numpy_tensor)
        tensorshare = TensorShare(
            tensors=base64.b64encode(_tensors),
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(_tensors)
        assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_paddle_tensor")
    def test_tensorshare_with_paddle(self, dict_zeros_paddle_tensor) -> None:
        """Test the tensorshare schema with paddle tensors."""
        _tensors = serialize_paddle(dict_zeros_paddle_tensor)
        tensorshare = TensorShare(
            tensors=base64.b64encode(_tensors),
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(_tensors)
        assert tensorshare.size == len(_tensors)

        # @pytest.mark.usefixtures("dict_zeros_tensorflow_tensor")
        # def test_tensorshare_with_tensorflow(self, dict_zeros_tensorflow_tensor) -> None:
        """Test the tensorshare schema with tensorflow tensors."""
        # _tensors = serialize_tensorflow(dict_zeros_tensorflow_tensor)
        # tensorshare = TensorShare(
        #     tensors=base64.b64encode(_tensors),
        #     size=len(_tensors),
        # )

        # assert isinstance(tensorshare.tensors, bytes)
        # assert isinstance(tensorshare.size, int)

        # assert len(tensorshare.tensors) > 0
        # assert tensorshare.size > 0

        # assert tensorshare.tensors == base64.b64encode(_tensors)
        # assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_torch_tensor")
    def test_tensorshare_with_torch(self, dict_zeros_torch_tensor) -> None:
        """Test the tensorshare schema with torch tensors."""
        _tensors = serialize_torch(dict_zeros_torch_tensor)
        tensorshare = TensorShare(
            tensors=base64.b64encode(_tensors),
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(_tensors)
        assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_flax_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.FLAX, "flax"])
    def test_from_dict_method_with_flax(self, dict_zeros_flax_tensor, backend) -> None:
        """Test the from_dict method with flax tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_flax_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(
            serialize_flax(dict_zeros_flax_tensor)
        )
        assert tensorshare.size == ByteSize(len(serialize_flax(dict_zeros_flax_tensor)))

    @pytest.mark.usefixtures("dict_zeros_numpy_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.NUMPY, "numpy"])
    def test_from_dict_method_with_numpy(
        self, dict_zeros_numpy_tensor, backend
    ) -> None:
        """Test the from_dict method with numpy tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_numpy_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(
            serialize_numpy(dict_zeros_numpy_tensor)
        )
        assert tensorshare.size == ByteSize(
            len(serialize_numpy(dict_zeros_numpy_tensor))
        )

    @pytest.mark.usefixtures("dict_zeros_paddle_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_from_dict_method_with_paddle(
        self, dict_zeros_paddle_tensor, backend
    ) -> None:
        """Test the from_dict method with paddle tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_paddle_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

    # @pytest.mark.usefixtures("dict_zeros_tensorflow_tensor")
    # @parametrize("backend", [None, Backend.TENSORFLOW, "tensorflow"])
    # def test_from_dict_method_with_tensorflow(self, dict_zeros_tensorflow_tensor, backend) -> None:
    #     """Test the from_dict method with tensorflow tensors."""
    #     tensorshare = TensorShare.from_dict(dict_zeros_tensorflow_tensor, backend=backend)

    #     assert isinstance(tensorshare, TensorShare)
    #     assert isinstance(tensorshare.tensors, bytes)
    #     assert isinstance(tensorshare.size, int)

    #     assert len(tensorshare.tensors) > 0
    #     assert tensorshare.size > 0

    #     assert tensorshare.tensors == base64.b64encode(serialize_tensorflow(
    #         dict_zeros_tensorflow_tensor
    #     ))
    #     assert tensorshare.size == len(
    #         serialize_tensorflow(dict_zeros_tensorflow_tensor)
    #     )

    @pytest.mark.usefixtures("dict_zeros_torch_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.TORCH, "torch"])
    def test_from_dict_method_with_torch(
        self, dict_zeros_torch_tensor, backend
    ) -> None:
        """Test the from_dict method with torch tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_torch_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == base64.b64encode(
            serialize_torch(dict_zeros_torch_tensor)
        )
        assert tensorshare.size == ByteSize(
            len(serialize_torch(dict_zeros_torch_tensor))
        )

    # @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.usefixtures("dict_zeros_flax_tensor")
    @pytest.mark.parametrize("backend", [Backend.FLAX, "flax"])
    def test_to_tensors_method_with_flax(
        self,
        serialized_fixed_numpy_tensors,
        backend,
    ) -> None:
        """Test the to_tensors method with flax backend."""
        tensorshare = TensorShare(
            tensors=base64.b64encode(serialized_fixed_numpy_tensors),
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], jnp.ndarray)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.NUMPY, "numpy"])
    def test_to_tensors_method_with_numpy(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with numpy backend."""
        tensorshare = TensorShare(
            tensors=base64.b64encode(serialized_fixed_numpy_tensors),
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], np.ndarray)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_to_tensors_method_with_paddle(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with paddle backend."""
        tensorshare = TensorShare(
            tensors=base64.b64encode(serialized_fixed_numpy_tensors),
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], paddle.Tensor)
        assert tensors["embeddings"].shape == [2, 2]

    # @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    # @pytest.mark.parametrize("backend", [Backend.TENSORFLOW, "tensorflow"])
    # def test_to_tensors_method_with_tensorflow(self, serialized_fixed_numpy_tensors, backend) -> None:
    #     """Test the to_tensors method with tensorflow backend."""
    #     tensorshare = TensorShare(
    #         tensors=base64.b64encode(serialized_fixed_numpy_tensors),
    #         size=len(serialized_fixed_numpy_tensors)
    #     )
    #     tensors = tensorshare.to_tensors(backend=backend)

    #     assert isinstance(tensors, dict)
    #     assert isinstance(tensors["embeddings"], tf.Tensor)
    #     assert tensors["embeddings"].shape == [2, 2]

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.TORCH, "torch"])
    def test_to_tensors_method_with_torch(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with torch backend."""
        tensorshare = TensorShare(
            tensors=base64.b64encode(serialized_fixed_numpy_tensors),
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], torch.Tensor)
        assert tensors["embeddings"].shape == torch.Size([2, 2])


class TestDefaultResponse:
    """Test the pydantic schema DefaultResponse."""

    def test_default_response_init(self) -> None:
        """Test the schema init."""
        response = DefaultResponse(message="Test")

        assert isinstance(response.message, str)
        assert response.message == "Test"
        assert response.model_dump() == {"message": "Test"}


class MockResponse(BaseModel):
    """Mock response."""

    mock: str


class TestTensorShareServer:
    """Test the pydantic schema TensorShareServer."""

    def test_schema_init(self) -> None:
        """Test the schema init."""
        server = TensorShareServer(
            url="https://localhost:8000",
            ping="https://localhost:8000/ping",
            receive_tensor="https://localhost:8000/receive_tensor",
        )

        assert server.response_model == DefaultResponse

        assert server.url == HttpUrl("https://localhost:8000")
        assert str(server.url) == "https://localhost:8000/"
        assert server.ping == HttpUrl("https://localhost:8000/ping")
        assert str(server.ping) == "https://localhost:8000/ping"
        assert server.receive_tensor == HttpUrl("https://localhost:8000/receive_tensor")
        assert str(server.receive_tensor) == "https://localhost:8000/receive_tensor"

        assert server.model_dump() == {
            "url": HttpUrl("https://localhost:8000"),
            "ping": HttpUrl("https://localhost:8000/ping"),
            "receive_tensor": HttpUrl("https://localhost:8000/receive_tensor"),
        }

    def test_schema_with_mock_response(self) -> None:
        """Test the schema with mock response."""
        server = TensorShareServer(
            url="https://localhost:8000",
            ping="https://localhost:8000/ping",
            receive_tensor="https://localhost:8000/receive_tensor",
            response_model=MockResponse,
        )

        assert server.response_model == MockResponse

    def test_schema_validation_error(self) -> None:
        """Test the schema validation error."""
        pydantic_version = pkg_resources.get_distribution("pydantic").version[:-2]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "1 validation error for TensorShareServer\nurl\n  URL scheme should be"
                " 'http' or 'https' [type=url_scheme, input_value='localhost:8000',"
                " input_type=str]\n    For further information visit"
                f" https://errors.pydantic.dev/{pydantic_version}/v/url_scheme"
            ),
        ):
            TensorShareServer(
                url="localhost:8000",
                ping="https://localhost:8000/ping",
                receive_tensor="https://localhost:8000/receive_tensor",
            )

    @pytest.mark.parametrize(
        "config",
        [
            {
                "url": "https://localhost:1234",
                "ping": "new_ping",
                "receive_tensor": "get_tensor",
                "response_model": MockResponse,
            },
            {
                "url": "https://localhost:8888",
                "receive_tensor": "get_tensor",
            },
            {
                "url": "https://localhost:9876",
                "ping": "new_ping",
            },
            {
                "url": "https://localhost:8091",
                "response_model": MockResponse,
            },
        ],
    )
    def test_schema_from_dict_method_with_custom(self, config: dict) -> None:
        """Test the schema from_dict method with custom."""
        server_config = TensorShareServer.from_dict(server_config=config)

        assert isinstance(server_config, TensorShareServer)

        assert server_config.url == HttpUrl(config["url"])
        assert str(server_config.url) == f"{config['url']}/"

        if "ping" in config:
            assert server_config.ping == HttpUrl(f"{config['url']}/{config['ping']}")
            assert str(server_config.ping) == f"{config['url']}/{config['ping']}"
        else:
            assert server_config.ping == HttpUrl(f"{config['url']}/ping")
            assert str(server_config.ping) == f"{config['url']}/ping"

        if "receive_tensor" in config:
            assert server_config.receive_tensor == HttpUrl(
                f"{config['url']}/{config['receive_tensor']}"
            )
            assert (
                str(server_config.receive_tensor)
                == f"{config['url']}/{config['receive_tensor']}"
            )
        else:
            assert server_config.receive_tensor == HttpUrl(
                f"{config['url']}/receive_tensor"
            )
            assert (
                str(server_config.receive_tensor) == f"{config['url']}/receive_tensor"
            )

        if "response_model" in config:
            assert server_config.response_model == MockResponse
        else:
            assert server_config.response_model == DefaultResponse

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"url": 123},
            {"ping": "https://localhost:8000/ping"},
            {"receive_tensor": "https://localhost:8000/receive_tensor"},
            {"response_model": DefaultResponse},
        ],
    )
    def test_schema_from_dict_method_with_invalid_dict(self, config: dict) -> None:
        """Test the schema from_dict method with invalid dict."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Verify the configuration dictionary, `url` is missing or invalid."
            ),
        ):
            TensorShareServer.from_dict(server_config=config)

    @pytest.mark.parametrize(
        ["target", "config"],
        [
            ("ping", {"url": "https://localhost:8000", "ping": 123}),
            (
                "receive_tensor",
                {"url": "https://localhost:8000", "receive_tensor": 123},
            ),
            (
                "response_model",
                {"url": "https://localhost:8000", "response_model": 123},
            ),
        ],
    )
    def test_schema_from_dict_method_with_invalid_value(
        self, target: str, config: dict
    ) -> None:
        """Test the schema from_dict method with invalid value."""
        if target == "response_model":
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "Verify the configuration dictionary, `response_model` must be a"
                    " subclass of BaseModel."
                ),
            ):
                TensorShareServer.from_dict(server_config=config)
        else:
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"Verify the configuration dictionary, `{target}` must be a string."
                ),
            ):
                TensorShareServer.from_dict(server_config=config)
