"""Test the tensorshare.serialization.constants module."""

import jaxlib
import numpy as np
import paddle

# import tensorflow as tf
import torch

from tensorshare.serialization.constants import (
    BACKEND_DESER_FUNC_MAPPING,
    BACKEND_SER_FUNC_MAPPING,
    TENSOR_TYPE_MAPPING,
    Backend,
)
from tensorshare.serialization.utils import (
    deserialize_flax,
    deserialize_numpy,
    deserialize_paddle,
    # deserialize_tensorflow,
    deserialize_torch,
    serialize_flax,
    serialize_numpy,
    serialize_paddle,
    # serialize_tensorflow,
    serialize_torch,
)


class TestBackendEnum:
    """Tests for the backend enum."""

    def test_backend_enum(self) -> None:
        """Test the backend enum."""
        assert len(Backend) == 4
        assert Backend.FLAX == "flax"
        assert Backend.NUMPY == "numpy"
        assert Backend.PADDLEPADDLE == "paddlepaddle"
        # assert Backend.TENSORFLOW == "tensorflow"
        assert Backend.TORCH == "torch"


class TestProcessorConstants:
    """Test the backend enum and associated constants."""

    def test_backend_ser_func_mapping(self) -> None:
        """Test the backends function mapping."""
        assert isinstance(BACKEND_SER_FUNC_MAPPING, dict)
        assert len(BACKEND_SER_FUNC_MAPPING) > 0

        assert "flax" in BACKEND_SER_FUNC_MAPPING
        assert "numpy" in BACKEND_SER_FUNC_MAPPING
        assert "paddlepaddle" in BACKEND_SER_FUNC_MAPPING
        # assert "tensorflow" in BACKEND_SER_FUNC_MAPPING
        assert "torch" in BACKEND_SER_FUNC_MAPPING

        assert BACKEND_SER_FUNC_MAPPING["flax"] == serialize_flax
        assert BACKEND_SER_FUNC_MAPPING["numpy"] == serialize_numpy
        assert BACKEND_SER_FUNC_MAPPING["paddlepaddle"] == serialize_paddle
        # assert BACKEND_SER_FUNC_MAPPING["tensorflow"] == serialize_tensorflow
        assert BACKEND_SER_FUNC_MAPPING["torch"] == serialize_torch

    def test_backend_deser_func_mapping(self) -> None:
        """Test the backend deserialization function mapping."""
        assert isinstance(BACKEND_DESER_FUNC_MAPPING, dict)
        assert len(BACKEND_DESER_FUNC_MAPPING) > 0

        assert "flax" in BACKEND_DESER_FUNC_MAPPING
        assert "numpy" in BACKEND_DESER_FUNC_MAPPING
        assert "paddlepaddle" in BACKEND_DESER_FUNC_MAPPING
        # assert "tensorflow" in BACKEND_DESER_FUNC_MAPPING
        assert "torch" in BACKEND_DESER_FUNC_MAPPING

        assert BACKEND_DESER_FUNC_MAPPING["flax"] == deserialize_flax
        assert BACKEND_DESER_FUNC_MAPPING["numpy"] == deserialize_numpy
        assert BACKEND_DESER_FUNC_MAPPING["paddlepaddle"] == deserialize_paddle
        # assert BACKEND_DESER_FUNC_MAPPING["tensorflow"] == deserialize_tensorflow
        assert BACKEND_DESER_FUNC_MAPPING["torch"] == deserialize_torch

    def test_tensor_type_mapping(self) -> None:
        """Test the tensor type mapping."""
        assert isinstance(TENSOR_TYPE_MAPPING, dict)
        assert len(TENSOR_TYPE_MAPPING) > 0

        assert jaxlib.xla_extension.ArrayImpl in TENSOR_TYPE_MAPPING
        assert np.ndarray in TENSOR_TYPE_MAPPING
        assert paddle.Tensor in TENSOR_TYPE_MAPPING
        # assert tf.Tensor in TENSOR_TYPE_MAPPING
        assert torch.Tensor in TENSOR_TYPE_MAPPING

        assert TENSOR_TYPE_MAPPING[jaxlib.xla_extension.ArrayImpl] == "flax"
        assert TENSOR_TYPE_MAPPING[np.ndarray] == "numpy"
        assert TENSOR_TYPE_MAPPING[paddle.Tensor] == "paddlepaddle"
        # assert TENSOR_TYPE_MAPPING[tf.Tensor] == "tensorflow"
        assert TENSOR_TYPE_MAPPING[torch.Tensor] == "torch"
