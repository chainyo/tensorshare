"""Test the TensorProcessor interface and the associated functions."""

import base64
import re

import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch
from jax import Array
from safetensors import SafetensorError

from tensorshare.serialization.constants import (
    BACKEND_TENSOR_TYPE_MAPPING,
    Backend,
    TensorType,
)
from tensorshare.serialization.processor import (
    TensorProcessor,
    _infer_backend,
)


class TestTensorProcessorSerialize:
    """Test the TensorProcessor serialize method"""

    @pytest.mark.usefixtures("multiple_flax_tensors")
    def test_infer_backend_with_multiple_flax_tensors(
        self, multiple_flax_tensors
    ) -> None:
        """Test the _infer_backend function with multiple flax tensors."""
        tensor_type = type(multiple_flax_tensors["embeddings"])
        type_str = f"{tensor_type.__module__}.{tensor_type.__name__}"
        backend = _infer_backend(multiple_flax_tensors)

        assert isinstance(backend, str)
        assert backend == "flax"
        assert backend == Backend.FLAX
        assert backend == BACKEND_TENSOR_TYPE_MAPPING[type_str]

    @pytest.mark.usefixtures("multiple_numpy_tensors")
    def test_infer_backend_with_multiple_numpy_tensors(
        self, multiple_numpy_tensors
    ) -> None:
        """Test the _infer_backend function with multiple numpy tensors."""
        tensor_type = type(multiple_numpy_tensors["embeddings"])
        type_str = f"{tensor_type.__module__}.{tensor_type.__name__}"
        backend = _infer_backend(multiple_numpy_tensors)

        assert isinstance(backend, str)
        assert backend == "numpy"
        assert backend == Backend.NUMPY
        assert backend == BACKEND_TENSOR_TYPE_MAPPING[type_str]

    @pytest.mark.usefixtures("multiple_paddle_tensors")
    def test_infer_backend_with_multiple_paddle_tensors(
        self, multiple_paddle_tensors
    ) -> None:
        """Test the _infer_backend function with multiple paddle tensors."""
        tensor_type = type(multiple_paddle_tensors["embeddings"])
        type_str = f"{tensor_type.__module__}.{tensor_type.__name__}"
        backend = _infer_backend(multiple_paddle_tensors)

        assert isinstance(backend, str)
        assert backend == "paddlepaddle"
        assert backend == Backend.PADDLEPADDLE
        assert backend == BACKEND_TENSOR_TYPE_MAPPING[type_str]

    # @pytest.mark.usefixtures("multiple_tensorflow_tensors")
    # def test_infer_backend_with_multiple_tensorflow_tensors(
    #     self, multiple_tensorflow_tensors
    # ) -> None:
    #     """Test the _infer_backend function with multiple tensorflow tensors."""
    #     tensor_type = type(multiple_tensorflow_tensors["embeddings"])
    #     type_str = f"{tensor_type.__module__}.{tensor_type.__name__}"
    #     backend = _infer_backend(multiple_tensorflow_tensors)

    #     assert isinstance(backend, str)
    #     assert backend == "tensorflow"
    #     assert backend == Backend.TENSORFLOW
    #     assert backend == BACKEND_TENSOR_TYPE_MAPPING[type_str]

    @pytest.mark.usefixtures("multiple_torch_tensors")
    def test_infer_backend_with_multiple_torch_tensors(
        self, multiple_torch_tensors
    ) -> None:
        """Test the _infer_backend function with multiple torch tensors."""
        tensor_type = type(multiple_torch_tensors["embeddings"])
        type_str = f"{tensor_type.__module__}.{tensor_type.__name__}"
        backend = _infer_backend(multiple_torch_tensors)

        assert isinstance(backend, str)
        assert backend == "torch"
        assert backend == Backend.TORCH
        assert backend == BACKEND_TENSOR_TYPE_MAPPING[type_str]

    @pytest.mark.usefixtures("multiple_backend_tensors")
    def test_infer_backend_with_multiple_backend_tensors(
        self, multiple_backend_tensors
    ) -> None:
        """Test the _infer_backend function with multiple backend tensors."""
        tensor_types = [
            f"{type(tensor).__module__}.{type(tensor).__name__}"
            for tensor in multiple_backend_tensors.values()
        ]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"All tensors must have the same type, got {tensor_types} instead."
            ),
        ):
            _infer_backend(multiple_backend_tensors)

    def test_infer_backend_with_unsupported_tensor(self) -> None:
        """Test the serialize function with an unsupported tensor."""
        tensors = {"unsupported": 1}
        first_tensor_type = [
            f"{type(tensor).__module__}.{type(tensor).__name__}"
            for tensor in tensors.values()
        ][0]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Unsupported tensor type {first_tensor_type}. Supported types are"
                f" {list(TensorType)}\nThe supported backends are"
                f" {list(Backend)}."
            ),
        ):
            TensorProcessor.serialize(tensors)

    @pytest.mark.usefixtures("multiple_flax_tensors")
    @pytest.mark.parametrize("backend", [None, Backend.FLAX, "flax"])
    def test_tensor_processor_serialize_with_multiple_flax_tensors_and_backend(
        self, multiple_flax_tensors, backend
    ) -> None:
        """Test the serialize function with multiple flax tensors and backend."""
        tensorshare = TensorProcessor.serialize(multiple_flax_tensors, backend=backend)

        assert isinstance(tensorshare, tuple)
        assert isinstance(tensorshare[0], bytes)
        assert isinstance(tensorshare[1], int)

    @pytest.mark.usefixtures("multiple_numpy_tensors")
    @pytest.mark.parametrize("backend", [None, Backend.NUMPY, "numpy"])
    def test_tensor_processor_serialize_with_multiple_numpy_tensors_and_backend(
        self, multiple_numpy_tensors, backend
    ) -> None:
        """Test the serialize function with multiple numpy tensors and backend."""
        tensorshare = TensorProcessor.serialize(multiple_numpy_tensors, backend=backend)

        assert isinstance(tensorshare, tuple)
        assert isinstance(tensorshare[0], bytes)
        assert isinstance(tensorshare[1], int)

    @pytest.mark.usefixtures("multiple_paddle_tensors")
    @pytest.mark.parametrize("backend", [None, Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_tensor_processor_serialize_with_multiple_paddle_tensors_and_backend(
        self, multiple_paddle_tensors, backend
    ) -> None:
        """Test the serialize function with multiple paddle tensors and backend."""
        tensorshare = TensorProcessor.serialize(
            multiple_paddle_tensors, backend=backend
        )

        assert isinstance(tensorshare, tuple)
        assert isinstance(tensorshare[0], bytes)
        assert isinstance(tensorshare[1], int)

    # @pytest.mark.usefixtures("multiple_tensorflow_tensors")
    # @parametrize("backend", [None, Backend.TENSORFLOW, "tensorflow"])
    # def test_tensor_processor_serialize_with_multiple_tensorflow_tensors_and_backend(
    #     self, multiple_tensorflow_tensors, backend
    # ) -> None:
    #     """Test the serialize function with multiple tensorflow tensors and backend."""
    #     tensorshare = TensorProcessor.serialize(multiple_tensorflow_tensors, backend=backend)

    # assert isinstance(tensorshare, tuple)
    # assert isinstance(tensorshare[0], bytes)
    # assert isinstance(tensorshare[1], int)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    @pytest.mark.parametrize("backend", [None, Backend.TORCH, "torch"])
    def test_tensor_processor_serialize_with_multiple_torch_tensors_and_backend(
        self, multiple_torch_tensors, backend
    ) -> None:
        """Test the serialize function with multiple torch tensors and backend."""
        tensorshare = TensorProcessor.serialize(multiple_torch_tensors, backend=backend)

        assert isinstance(tensorshare, tuple)
        assert isinstance(tensorshare[0], bytes)
        assert isinstance(tensorshare[1], int)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    def test_tensor_processor_serialize_with_metadata(
        self, multiple_torch_tensors
    ) -> None:
        """Test the serialize function with metadata."""
        metadata = {"foo": "bar"}
        tensorshare = TensorProcessor.serialize(
            multiple_torch_tensors, metadata=metadata
        )

        assert isinstance(tensorshare, tuple)
        assert isinstance(tensorshare[0], bytes)
        assert isinstance(tensorshare[1], int)

    @pytest.mark.usefixtures("multiple_backend_tensors")
    def test_tensor_processor_serialize_with_multiple_backend_tensors(
        self, multiple_backend_tensors
    ) -> None:
        """Test the serialize function with multiple backend tensors."""
        tensor_types = [
            f"{type(tensor).__module__}.{type(tensor).__name__}"
            for tensor in multiple_backend_tensors.values()
        ]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"All tensors must have the same type, got {tensor_types} instead."
            ),
        ):
            TensorProcessor.serialize(multiple_backend_tensors)

    def test_tensor_processor_serialize_with_empty_dict(self) -> None:
        """Test the serialize function with an empty dictionary."""
        with pytest.raises(
            ValueError,
            match=re.escape("Tensors dictionary cannot be empty."),
        ):
            TensorProcessor.serialize({})

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
    def test_tensor_processor_serialize_with_invalid_tensors_format(
        self, tensors
    ) -> None:
        """Test the serialize function with an invalid tensors format."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Tensors must be a dictionary, got `{type(tensors)}` instead."
            ),
        ):
            TensorProcessor.serialize(tensors)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    @pytest.mark.parametrize(
        "backend",
        ["invalid", "mxnet", "caffe2", "chainer"],
    )
    def test_tensor_processor_serialize_with_invalid_backend(
        self, multiple_torch_tensors, backend
    ) -> None:
        """Test the serialize function with an invalid backend."""
        with pytest.raises(
            KeyError,
            match=re.escape(
                f"Invalid backend `{backend}`. Must be one of {list(Backend)}."
            ),
        ):
            TensorProcessor.serialize(multiple_torch_tensors, backend=backend)

    @pytest.mark.usefixtures("multiple_torch_tensors")
    @pytest.mark.parametrize(
        "backend",
        [1, 1.0, ["backend"], ("backend",), {"backend": "torch"}],
    )
    def test_tensor_processor_serialize_with_invalid_backend_format(
        self, multiple_torch_tensors, backend
    ) -> None:
        """Test the serialize function with an invalid backend format."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Backend must be a string or an instance of Backend enum, got"
                f" `{type(backend)}` instead. Use"
                " `tensorshare.serialization.Backend` to access the Backend enum."
                " If you don't specify a backend, it will be inferred from the"
                " tensors format."
            ),
        ):
            TensorProcessor.serialize(multiple_torch_tensors, backend=backend)


class TestTensorProcessorDeserialize:
    """Test the TensorProcessor deserialize method."""

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.FLAX, "flax"])
    def test_tensor_processor_deserialize_with_flax_backend(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the deserialize function with flax backend."""
        with pytest.raises(
            SafetensorError,
            match=re.escape("Error while deserializing: HeaderTooLarge"),
        ):
            TensorProcessor.deserialize(serialized_fixed_numpy_tensors, backend=backend)

        b64_encoded = base64.b64encode(serialized_fixed_numpy_tensors)
        tensors = TensorProcessor.deserialize(b64_encoded, backend=backend)

        assert isinstance(tensors, dict)
        assert len(tensors) > 0

        assert isinstance(tensors["embeddings"], Array)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.NUMPY, "numpy"])
    def test_tensor_processor_deserialize_with_numpy_backend(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the deserialize function with numpy backend."""
        with pytest.raises(
            SafetensorError,
            match=re.escape("Error while deserializing: HeaderTooLarge"),
        ):
            TensorProcessor.deserialize(serialized_fixed_numpy_tensors, backend=backend)

        b64_encoded = base64.b64encode(serialized_fixed_numpy_tensors)
        tensors = TensorProcessor.deserialize(b64_encoded, backend=backend)

        assert isinstance(tensors, dict)
        assert len(tensors) > 0

        assert isinstance(tensors["embeddings"], np.ndarray)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_tensor_processor_deserialize_with_paddlepaddle_backend(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the deserialize function with paddlepaddle backend."""
        with pytest.raises(
            SafetensorError,
            match=re.escape("Error while deserializing: HeaderTooLarge"),
        ):
            TensorProcessor.deserialize(serialized_fixed_numpy_tensors, backend=backend)

        b64_encoded = base64.b64encode(serialized_fixed_numpy_tensors)
        tensors = TensorProcessor.deserialize(b64_encoded, backend=backend)

        assert isinstance(tensors, dict)
        assert len(tensors) > 0

        assert isinstance(tensors["embeddings"], paddle.Tensor)
        assert tensors["embeddings"].shape == [2, 2]

    # @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    # @pytest.mark.parametrize("backend", [Backend.TENSORFLOW, "tensorflow"])
    # def test_tensor_processor_deserialize_with_tensorflow_backend(
    #     self, serialized_fixed_numpy_tensors, backend
    # ) -> None:
    #     """Test the deserialize function with tensorflow backend."""
    #     with pytest.raises(
    #         SafetensorError,
    #         match=re.escape(
    #             "Error while deserializing: HeaderTooLarge"
    #         ),
    #     ):
    #         TensorProcessor.deserialize(
    #             serialized_fixed_numpy_tensors, backend=backend
    #         )

    #     b64_encoded = base64.b64encode(serialized_fixed_numpy_tensors)
    #     tensors = TensorProcessor.deserialize(b64_encoded, backend=backend)

    #     assert isinstance(tensors, dict)
    #     assert len(tensors) > 0

    #     assert isinstance(tensors["embeddings"], tf.Tensor)
    #     assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.TORCH, "torch"])
    def test_tensor_processor_deserialize_with_torch_backend(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the deserialize function with torch backend."""
        with pytest.raises(
            SafetensorError,
            match=re.escape("Error while deserializing: HeaderTooLarge"),
        ):
            TensorProcessor.deserialize(serialized_fixed_numpy_tensors, backend=backend)

        b64_encoded = base64.b64encode(serialized_fixed_numpy_tensors)
        tensors = TensorProcessor.deserialize(b64_encoded, backend=backend)

        assert isinstance(tensors, dict)
        assert len(tensors) > 0

        assert isinstance(tensors["embeddings"], torch.Tensor)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.parametrize(
        "data",
        [
            1,
            1.0,
            "tensors",
            ["tensors"],
            ("tensors",),
            {"tensors": "tensors"},
        ],
    )
    def test_tensor_processor_deserialize_with_invalid_data_format(self, data) -> None:
        """Test the deserialize function with an invalid data format."""
        with pytest.raises(
            TypeError,
            match=re.escape(f"Data must be bytes, got `{type(data)}` instead."),
        ):
            TensorProcessor.deserialize(data, backend=Backend.TORCH)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize(
        "backend",
        ["invalid", "mxnet", "caffe2", "chainer"],
    )
    def test_tensor_processor_deserialize_with_invalid_backend(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the deserialize function with an invalid backend."""
        with pytest.raises(
            KeyError,
            match=re.escape(
                f"Invalid backend `{backend}`. Must be one of {list(Backend)}."
            ),
        ):
            TensorProcessor.deserialize(serialized_fixed_numpy_tensors, backend=backend)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize(
        "backend",
        [1, 1.0, ["backend"], ("backend",), {"backend": "torch"}],
    )
    def test_tensor_processor_deserialize_with_invalid_backend_format(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the deserialize function with an invalid backend format."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Backend must be a string or an instance of Backend enum, got"
                f" `{type(backend)}` instead. Use `tensorshare.serialization.Backend`"
                " to access the Backend enum."
            ),
        ):
            TensorProcessor.deserialize(serialized_fixed_numpy_tensors, backend=backend)
