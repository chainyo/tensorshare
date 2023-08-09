"""Test utils functions for the serialization module."""

import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch
from jax import Array

from tensorshare.serialization.flax import (
    deserialize as flax_deserialize,
)
from tensorshare.serialization.flax import (
    serialize as flax_serialize,
)
from tensorshare.serialization.numpy import (
    deserialize as numpy_deserialize,
)
from tensorshare.serialization.numpy import (
    serialize as numpy_serialize,
)
from tensorshare.serialization.paddle import (
    deserialize as paddle_deserialize,
)
from tensorshare.serialization.paddle import (
    serialize as paddle_serialize,
)

# from tensorshare.serialization.tensorflow import (
#     deserialize as tf_deserialize,
#     serialize as tf_serialize,
# )
from tensorshare.serialization.torch import (
    deserialize as torch_deserialize,
)
from tensorshare.serialization.torch import (
    serialize as torch_serialize,
)


class TestSerializationFunctions:
    """Test serialization functions for the serialization module."""

    @pytest.mark.usefixtures("zeros_flax_tensor")
    def test_serialize_flax(self, zeros_flax_tensor) -> None:
        """Test convertion from flax to safetensors."""
        _tensors = {"embeddings": zeros_flax_tensor}
        tensor_bytes = flax_serialize(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_numpy_tensor")
    def test_serialize_numpy(self, zeros_numpy_tensor) -> None:
        """Test convertion from numpy to safetensors."""
        _tensors = {"embeddings": zeros_numpy_tensor}
        tensor_bytes = numpy_serialize(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_paddle_tensor")
    def test_serialize_paddle(self, zeros_paddle_tensor) -> None:
        """Test convertion from paddle to safetensors."""
        _tensors = {"embeddings": zeros_paddle_tensor}
        tensor_bytes = paddle_serialize(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    # @pytest.mark.usefixtures("zeros_tensorflow_tensor")
    # def test_serialize_tensorflow(self, zeros_tensorflow_tensor) -> None:
    #     """Test convertion from tensorflow to safetensors."""
    #     tensor_bytes = tensorflow_serialize(zeros_tensorflow_tensor)
    #
    #     assert isinstance(tensor_bytes, bytes)
    #     assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_torch_tensor")
    def test_serialize_torch(self, zeros_torch_tensor) -> None:
        """Test convertion from torch to safetensors."""
        _tensors = {"embeddings": zeros_torch_tensor}
        tensor_bytes = torch_serialize(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0


class TestDeserializationFunctions:
    """Test deserialization functions for the serialization module."""

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_numpy(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to numpy."""
        _tensors = numpy_deserialize(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], np.ndarray)
        assert _tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_flax(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to flax."""
        _tensors = flax_deserialize(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], Array)
        assert _tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_paddle(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to paddle."""
        _tensors = paddle_deserialize(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], paddle.Tensor)
        assert _tensors["embeddings"].shape == [2, 2]

    # @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    # def test_deserialize_tensorflow(self, serialized_fixed_numpy_tensors) -> None:
    #     """Test convertion from safetensors to tensorflow."""
    #     _tensors = tensorflow_deserialize(serialized_fixed_numpy_tensors)
    #
    #     assert isinstance(_tensors, dict)
    #     assert len(_tensors) > 0
    #
    #     assert isinstance(_tensors["embeddings"], tf.Tensor)
    #     assert _tensors["embeddings"].shape == [2, 2]

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_deserialize_torch(self, serialized_fixed_numpy_tensors) -> None:
        """Test convertion from safetensors to torch."""
        _tensors = torch_deserialize(serialized_fixed_numpy_tensors)

        assert isinstance(_tensors, dict)
        assert len(_tensors) > 0

        assert isinstance(_tensors["embeddings"], torch.Tensor)
        assert _tensors["embeddings"].shape == (2, 2)
