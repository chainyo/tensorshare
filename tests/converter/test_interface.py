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
    _infer_backend,
    TensorConverter,
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
    def test_infer_backend_with_multiple_flax_tensors(
        self, multiple_flax_tensors
    ) -> None:
        """Test the _infer_backend function with multiple flax tensors."""
        tensor_type = type(multiple_flax_tensors["embeddings"])
        backend = _infer_backend(multiple_flax_tensors)

        assert isinstance(backend, str)
        assert backend == "flax"
        assert backend == Backend.FLAX
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_numpy_tensors")
    def test_infer_backend_with_multiple_numpy_tensors(
        self, multiple_numpy_tensors
    ) -> None:
        """Test the _infer_backend function with multiple numpy tensors."""
        tensor_type = type(multiple_numpy_tensors["embeddings"])
        backend = _infer_backend(multiple_numpy_tensors)

        assert isinstance(backend, str)
        assert backend == "numpy"
        assert backend == Backend.NUMPY
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_paddle_tensors")
    def test_infer_backend_with_multiple_paddle_tensors(
        self, multiple_paddle_tensors
    ) -> None:
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
    def test_infer_backend_with_multiple_torch_tensors(
        self, multiple_torch_tensors
    ) -> None:
        """Test the _infer_backend function with multiple torch tensors."""
        tensor_type = type(multiple_torch_tensors["embeddings"])
        backend = _infer_backend(multiple_torch_tensors)

        assert isinstance(backend, str)
        assert backend == "torch"
        assert backend == Backend.TORCH
        assert backend in BACKENDS_FUNC_MAPPING
        assert backend == TENSOR_TYPE_MAPPING[tensor_type]

    @pytest.mark.usefixtures("multiple_backend_tensors")
    def test_infer_backend_with_multiple_backend_tensors(
        self, multiple_backend_tensors
    ) -> None:
        """Test the _infer_backend function with multiple backend tensors."""
        tensor_types = [type(tensor) for tensor in multiple_backend_tensors.values()]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"All tensors must have the same type, got {tensor_types} instead."
            ),
        ):
            _infer_backend(multiple_backend_tensors)

    def test_infer_backend_with_unsupported_tensor(self) -> None:
        """Test the convert function with an unsupported tensor."""
        tensors = {"unsupported": 1} 
        first_tensor_type = [type(tensor) for tensor in tensors.values()][0]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Unsupported tensor type {first_tensor_type}. Supported types are"
                f" {list(TENSOR_TYPE_MAPPING.keys())}\nThe supported backends are"
                f" {list(BACKENDS_FUNC_MAPPING.keys())}."
            ),
        ):
            TensorConverter.convert(tensors)

    @pytest.mark.usefixtures("multiple_flax_tensors")
    def test_tensor_converter_convert_with_multiple_flax_tensors(self, multiple_flax_tensors) -> None:
        """Test the convert function with multiple flax tensors."""
        tensorshare = TensorConverter.convert(multiple_flax_tensors)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_flax_tensors")
    @pytest.mark.parametrize("backend", [Backend.FLAX, "flax"])
    def test_tensor_converter_convert_with_multiple_flax_tensors_and_backend(
        self, multiple_flax_tensors, backend
    ) -> None:
        """Test the convert function with multiple flax tensors and backend."""
        tensorshare = TensorConverter.convert(multiple_flax_tensors, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_numpy_tensors")
    def test_tensor_converter_convert_with_multiple_numpy_tensors(self, multiple_numpy_tensors) -> None:
        """Test the convert function with multiple numpy tensors."""
        tensorshare = TensorConverter.convert(multiple_numpy_tensors)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.NUMPY, "numpy"])
    def test_tensor_converter_convert_with_multiple_numpy_tensors_and_backend(
        self, multiple_numpy_tensors, backend
    ) -> None:
        """Test the convert function with multiple numpy tensors and backend."""
        tensorshare = TensorConverter.convert(multiple_numpy_tensors, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_paddle_tensors")
    def test_tensor_converter_convert_with_multiple_paddle_tensors(self, multiple_paddle_tensors) -> None:
        """Test the convert function with multiple paddle tensors."""
        tensorshare = TensorConverter.convert(multiple_paddle_tensors)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_paddle_tensors")
    @pytest.mark.parametrize("backend", [Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_tensor_converter_convert_with_multiple_paddle_tensors_and_backend(
        self, multiple_paddle_tensors, backend
    ) -> None:
        """Test the convert function with multiple paddle tensors and backend."""
        tensorshare = TensorConverter.convert(multiple_paddle_tensors, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    # @pytest.mark.usefixtures("multiple_tensorflow_tensors")
    # def test_tensor_converter_convert_with_multiple_tensorflow_tensors(self, multiple_tensorflow_tensors) -> None:
    #     """Test the convert function with multiple tensorflow tensors."""
    #     tensorshare = TensorConverter.convert(multiple_tensorflow_tensors)

    #     assert isinstance(tensorshare, TensorShare)
    #     assert isinstance(tensorshare.tensors, bytes)
    #     assert isinstance(tensorshare.size, int)
    #     assert isinstance(tensorshare.size_as_str, str)

    # @pytest.mark.usefixtures("multiple_tensorflow_tensors")
    # @parametrize("backend", [Backend.TENSORFLOW, "tensorflow"])
    # def test_tensor_converter_convert_with_multiple_tensorflow_tensors_and_backend(
    #     self, multiple_tensorflow_tensors, backend
    # ) -> None:
    #     """Test the convert function with multiple tensorflow tensors and backend."""
    #     tensorshare = TensorConverter.convert(multiple_tensorflow_tensors, backend=backend)

    #     assert isinstance(tensorshare, TensorShare)
    #     assert isinstance(tensorshare.tensors, bytes)
    #     assert isinstance(tensorshare.size, int)
    #     assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    def test_tensor_converter_convert_with_multiple_torch_tensors(self, multiple_torch_tensors) -> None:
        """Test the convert function with multiple torch tensors."""
        tensorshare = TensorConverter.convert(multiple_torch_tensors)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    @pytest.mark.parametrize("backend", [Backend.TORCH, "torch"])
    def test_tensor_converter_convert_with_multiple_torch_tensors_and_backend(
        self, multiple_torch_tensors, backend
    ) -> None:
        """Test the convert function with multiple torch tensors and backend."""
        tensorshare = TensorConverter.convert(multiple_torch_tensors, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    def test_tensor_converter_convert_with_metadata(self, multiple_torch_tensors) -> None:
        """Test the convert function with metadata."""
        metadata = {"foo": "bar"}
        tensorshare = TensorConverter.convert(multiple_torch_tensors, metadata=metadata)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)
        assert isinstance(tensorshare.size_as_str, str)

    @pytest.mark.usefixtures("multiple_backend_tensors")
    def test_tensor_converter_convert_with_multiple_backend_tensors(self, multiple_backend_tensors) -> None:
        """Test the convert function with multiple backend tensors."""
        tensor_types = [type(tensor) for tensor in multiple_backend_tensors.values()]
        
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"All tensors must have the same type, got {tensor_types} instead."
            ),
        ):
            TensorConverter.convert(multiple_backend_tensors)

    def test_tensor_converter_convert_with_empty_dict(self) -> None:
        """Test the convert function with an empty dict."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tensors dictionary cannot be empty."
            ),
        ):
            TensorConverter.convert({})

    @pytest.mark.parametrize(
        "tensors",
        [
            1,
            1.0,
            "tensors",
            ["tensors"],
            ("tensors",),
        ],
    )
    def test_tensor_converter_convert_with_invalid_tensors_format(self, tensors) -> None:
        """Test the convert function with an invalid tensors format."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Tensors must be a dictionary, got `{type(tensors)}` instead."
            ),
        ):
            TensorConverter.convert(tensors)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    @pytest.mark.parametrize(
        "backend", ["invalid", "mxnet", "caffe2", "chainer"],
    )
    def test_tensor_converter_convert_with_invalid_backend(
        self, multiple_torch_tensors, backend
    ) -> None:
        """Test the convert function with an invalid backend."""
        with pytest.raises(
            KeyError,
            match=re.escape(
                f"Invalid backend `{backend}`. Must be one of {list(Backend.__members__)}."
            ),
        ):
            TensorConverter.convert(multiple_torch_tensors, backend=backend)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    @pytest.mark.parametrize(
        "backend", [1, 1.0, ["backend"], ("backend",), {"backend": "torch"}],
    )
    def test_tensor_converter_convert_with_invalid_backend_format(
        self, multiple_torch_tensors, backend
    ) -> None:
        """Test the convert function with an invalid backend format."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Backend must be a string or an instance of Backend enum, got `{type(backend)}` instead. "
                "Use `tensorshare.schema.Backend` to access the Backend enum. "
                "If you don't specify a backend, it will be inferred from the tensors format."
            ),
        ):
            TensorConverter.convert(multiple_torch_tensors, backend=backend)
