"""Test utils functions for the serialization module."""

import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch
from jax import Array

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


class TestSerializationFunctions:
    """Test serialization functions for the serialization module."""

    @pytest.mark.usefixtures("zeros_flax_tensor")
    def test_serialize_flax(self, zeros_flax_tensor) -> None:
        """Test convertion from flax to safetensors."""
        _tensors = {"embeddings": zeros_flax_tensor}
        tensor_bytes = serialize_flax(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_numpy_tensor")
    def test_serialize_numpy(self, zeros_numpy_tensor) -> None:
        """Test convertion from numpy to safetensors."""
        _tensors = {"embeddings": zeros_numpy_tensor}
        tensor_bytes = serialize_numpy(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_paddle_tensor")
    def test_serialize_paddle(self, zeros_paddle_tensor) -> None:
        """Test convertion from paddle to safetensors."""
        _tensors = {"embeddings": zeros_paddle_tensor}
        tensor_bytes = serialize_paddle(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    # @pytest.mark.usefixtures("zeros_tensorflow_tensor")
    # def test_serialize_tensorflow(self, zeros_tensorflow_tensor) -> None:
    #     """Test convertion from tensorflow to safetensors."""
    #     tensor_bytes = serialize_tensorflow(zeros_tensorflow_tensor)
    #
    #     assert isinstance(tensor_bytes, bytes)
    #     assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_torch_tensor")
    def test_serialize_torch(self, zeros_torch_tensor) -> None:
        """Test convertion from torch to safetensors."""
        _tensors = {"embeddings": zeros_torch_tensor}
        tensor_bytes = serialize_torch(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0


class TestDeserializationFunctions:
    """Test deserialization functions for the serialization module."""

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_numpy(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to numpy."""
        _tensors = deserialize_numpy(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], np.ndarray)
        assert _tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_flax(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to flax."""
        _tensors = deserialize_flax(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], Array)
        assert _tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_paddle(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to paddle."""
        _tensors = deserialize_paddle(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], paddle.Tensor)
        assert _tensors["embeddings"].shape == [2, 2]

    # @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    # def test_deserialize_tensorflow(self, serialized_fixed_numpy_tensors) -> None:
    #     """Test convertion from safetensors to tensorflow."""
    #     _tensors = deserialize_tensorflow(serialized_fixed_numpy_tensors)
    #
    #     assert isinstance(_tensors, dict)
    #     assert len(_tensors) > 0
    #
    #     assert isinstance(_tensors["embeddings"], tf.Tensor)
    #     assert _tensors["embeddings"].shape == [2, 2]

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_torch(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to torch."""
        _tensors = deserialize_torch(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], torch.Tensor)
        assert _tensors["embeddings"].shape == (2, 2)
