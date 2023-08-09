"""Constants for tensorshare.serialization."""

from enum import Enum
from typing import Dict, OrderedDict


class Backend(str, Enum):
    """Supported backends."""

    FLAX = "flax"
    NUMPY = "numpy"
    PADDLEPADDLE = "paddlepaddle"
    # TENSORFLOW = "tensorflow"
    TORCH = "torch"


class TensorType(str, Enum):
    """Tensor types."""

    FLAX = "jaxlib.xla_extension.ArrayImpl"
    NUMPY = "numpy.ndarray"
    PADDLEPADDLE = "paddle.Tensor"
    # TENSORFLOW = "tensorflow.Tensor"
    TORCH = "torch.Tensor"


# Mapping between backend and serialization module
BACKEND_MODULE_MAPPING: Dict[Backend, str] = OrderedDict(
    [
        (Backend.FLAX, "tensorshare.serialization.flax"),
        (Backend.NUMPY, "tensorshare.serialization.numpy"),
        (Backend.PADDLEPADDLE, "tensorshare.serialization.paddle"),
        # (Backend.TENSORFLOW, "tensorshare.serialization.tensorflow"),
        (Backend.TORCH, "tensorshare.serialization.torch"),
    ]
)
# Mapping between backend and tensor type
BACKEND_TENSOR_TYPE_MAPPING: Dict[TensorType, Backend] = OrderedDict(
    [
        (TensorType.FLAX, Backend.FLAX),
        (TensorType.NUMPY, Backend.NUMPY),
        (TensorType.PADDLEPADDLE, Backend.PADDLEPADDLE),
        # (TensorType.TENSORFLOW, Backend.TENSORFLOW),
        (TensorType.TORCH, Backend.TORCH),
    ]
)
