"""Test utils functions for the serialization module."""

import pytest

from tensorshare.serialization.utils import (
    serialize_flax,
    serialize_numpy,
    serialize_paddle,
    # serialize_tensorflow,
    serialize_torch,
)


class TestConvertionUtilsFunction:
    """Test utils functions for the converter module."""

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
