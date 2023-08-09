"""Test the tensorshare.serialization.constants module."""


# import tensorflow as tf

from tensorshare.serialization.constants import (
    BACKEND_MODULE_MAPPING,
    BACKEND_TENSOR_TYPE_MAPPING,
    Backend,
    TensorType,
)

# from tensorshare.serialization.tensorflow import (
#     deserialize as tensorflow_deserialize,
#     serialize as tensorflow_serialize,
# )


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


class TestTensorTypeEnum:
    """Tests for the tensor type enum."""

    def test_tensor_type_enum(self) -> None:
        """Test the tensor type enum."""
        assert len(TensorType) == 4
        assert TensorType.FLAX == "jaxlib.xla_extension.ArrayImpl"
        assert TensorType.NUMPY == "numpy.ndarray"
        assert TensorType.PADDLEPADDLE == "paddle.Tensor"
        # assert TensorType.TENSORFLOW == "tf.Tensor"
        assert TensorType.TORCH == "torch.Tensor"


class TestSerializationConstants:
    """Test the backend enum and associated constants."""

    def test_backend_module_mapping(self) -> None:
        """Test the backend module mapping."""
        assert isinstance(BACKEND_MODULE_MAPPING, dict)
        assert len(BACKEND_MODULE_MAPPING) > 0

        assert "flax" in BACKEND_MODULE_MAPPING
        assert "numpy" in BACKEND_MODULE_MAPPING
        assert "paddlepaddle" in BACKEND_MODULE_MAPPING
        # assert "tensorflow" in BACKEND_MODULE_MAPPING
        assert "torch" in BACKEND_MODULE_MAPPING

        assert BACKEND_MODULE_MAPPING["flax"] == "tensorshare.serialization.flax"
        assert BACKEND_MODULE_MAPPING["numpy"] == "tensorshare.serialization.numpy"
        assert (
            BACKEND_MODULE_MAPPING["paddlepaddle"] == "tensorshare.serialization.paddle"
        )
        # assert (
        #     BACKEND_MODULE_MAPPING["tensorflow"]
        #     == "tensorshare.serialization.tensorflow"
        # )
        assert BACKEND_MODULE_MAPPING["torch"] == "tensorshare.serialization.torch"

    def test_tensor_type_mapping(self) -> None:
        """Test the tensor type mapping."""
        assert isinstance(BACKEND_TENSOR_TYPE_MAPPING, dict)
        assert len(BACKEND_TENSOR_TYPE_MAPPING) > 0

        assert "jaxlib.xla_extension.ArrayImpl" in BACKEND_TENSOR_TYPE_MAPPING
        assert "numpy.ndarray" in BACKEND_TENSOR_TYPE_MAPPING
        assert "paddle.Tensor" in BACKEND_TENSOR_TYPE_MAPPING
        # assert "tf.Tensor" in BACKEND_TENSOR_TYPE_MAPPING
        assert "torch.Tensor" in BACKEND_TENSOR_TYPE_MAPPING

        assert BACKEND_TENSOR_TYPE_MAPPING["jaxlib.xla_extension.ArrayImpl"] == "flax"
        assert BACKEND_TENSOR_TYPE_MAPPING["numpy.ndarray"] == "numpy"
        assert BACKEND_TENSOR_TYPE_MAPPING["paddle.Tensor"] == "paddlepaddle"
        # assert BACKEND_TENSOR_TYPE_MAPPING["tf.Tensor"] == "tensorflow"
        assert BACKEND_TENSOR_TYPE_MAPPING["torch.Tensor"] == "torch"
