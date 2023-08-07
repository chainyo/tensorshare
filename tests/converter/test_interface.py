"""Test the TensorConverter interface and the associated functions."""

import re

import jaxlib
import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch

from tensorshare.converter.interface import (
    BACKENDS_FUNC_MAPPING,
    TENSOR_TYPE_MAPPING,
    TensorConverter,
    _infer_backend,
)
from tensorshare.converter.utils import (
    convert_flax_to_safetensors,
    convert_numpy_to_safetensors,
    convert_paddle_to_safetensors,
    # convert_tensorflow_to_safetensors,
    convert_torch_to_safetensors,
)
from tensorshare.schema import Backend, TensorShare


class TestTensorConverter:
    """Test the TensorConverter interface and the associated functions."""

    def test_backends_func_mapping(self) -> None:
        """Test the backends function mapping."""
        assert isinstance(BACKENDS_FUNC_MAPPING, dict)
        assert len(BACKENDS_FUNC_MAPPING) > 0

        assert "flax" in BACKENDS_FUNC_MAPPING
        assert "numpy" in BACKENDS_FUNC_MAPPING
        assert "paddlepaddle" in BACKENDS_FUNC_MAPPING
        # assert "tensorflow" in BACKENDS_FUNC_MAPPING
        assert "torch" in BACKENDS_FUNC_MAPPING

        assert BACKENDS_FUNC_MAPPING["flax"] == convert_flax_to_safetensors
        assert BACKENDS_FUNC_MAPPING["numpy"] == convert_numpy_to_safetensors
        assert BACKENDS_FUNC_MAPPING["paddlepaddle"] == convert_paddle_to_safetensors
        # assert BACKENDS_FUNC_MAPPING["tensorflow"] == convert_tensorflow_to_safetensors
        assert BACKENDS_FUNC_MAPPING["torch"] == convert_torch_to_safetensors

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

    @pytest.mark.usefixtures("multiple_flax_tensors")
    def test_infer_backend_with_multiple_flax_tensors(self, multiple_flax_tensors) -> None:
        """Test the _infer_backend function with multiple flax tensors."""
        tensor_type = type(multiple_flax_tensors["embeddings"])
        backend = _infer_backend(multiple_flax_tensors)

        assert isinstance(backend, str)
        assert backend == "flax"
        assert backend == Backend.FLAX
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_numpy_tensors")
    def test_infer_backend_with_multiple_numpy_tensors(self, multiple_numpy_tensors) -> None:
        """Test the _infer_backend function with multiple numpy tensors."""
        tensor_type = type(multiple_numpy_tensors["embeddings"])
        backend = _infer_backend(multiple_numpy_tensors)

        assert isinstance(backend, str)
        assert backend == "numpy"
        assert backend == Backend.NUMPY
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_paddle_tensors")
    def test_infer_backend_with_multiple_paddle_tensors(self, multiple_paddle_tensors) -> None:
        """Test the _infer_backend function with multiple paddle tensors."""
        tensor_type = type(multiple_paddle_tensors["embeddings"])
        backend = _infer_backend(multiple_paddle_tensors)

        assert isinstance(backend, str)
        assert backend == "paddlepaddle"
        assert backend == Backend.PADDLEPADDLE
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    # @pytest.mark.usefixtures("multiple_tensorflow_tensors")
    # def test_infer_backend_with_multiple_tensorflow_tensors(
    #     self, multiple_tensorflow_tensors
    # ) -> None:
    #     """Test the _infer_backend function with multiple tensorflow tensors."""
    #     tensor_type = type(multiple_tensorflow_tensors["embeddings"])
    #     backend = _infer_backend(multiple_tensorflow_tensors)

    #     assert isinstance(backend, str)
    #     assert backend == "tensorflow"
    #     assert backend == Backend.TENSORFLOW
    #     assert backend in BACKENDS_FUNC_MAPPING
    #     assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_torch_tensors")
    def test_infer_backend_with_multiple_torch_tensors(self, multiple_torch_tensors) -> None:
        """Test the _infer_backend function with multiple torch tensors."""
        tensor_type = type(multiple_torch_tensors["embeddings"])
        backend = _infer_backend(multiple_torch_tensors)

        assert isinstance(backend, str)
        assert backend == "torch"
        assert backend == Backend.TORCH
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_backend_tensors")
    def test_infer_backend_with_multiple_backend_tensors(self, multiple_backend_tensors) -> None:
        """Test the _infer_backend function with multiple backend tensors."""
        tensor_types = [type(tensor) for tensor in multiple_backend_tensors.values()]

        with pytest.raises(ValueError, match=re.escape(f"All tensors must have the same type, got {tensor_types} instead.")):
            _infer_backend(multiple_backend_tensors)
