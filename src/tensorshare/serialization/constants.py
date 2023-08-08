"""Constants for tensorshare.serialization."""

from enum import Enum
from typing import Callable, Dict, OrderedDict, Union

import jaxlib
import numpy as np
import paddle

# import tensorflow as tf
import torch

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


class Backend(str, Enum):
    """Supported backends."""

    FLAX = "flax"
    NUMPY = "numpy"
    PADDLEPADDLE = "paddlepaddle"
    # TENSORFLOW = "tensorflow"
    TORCH = "torch"


# Mapping between backend and serialization function
BACKEND_SER_FUNC_MAPPING: Dict[Backend, Callable] = OrderedDict(
    [
        (Backend.FLAX, serialize_flax),
        (Backend.NUMPY, serialize_numpy),
        (Backend.PADDLEPADDLE, serialize_paddle),
        # (Backend.TENSORFLOW, serialize_tensorflow),
        (Backend.TORCH, serialize_torch),
    ]
)
# Mapping between backend and deserialization function
BACKEND_DESER_FUNC_MAPPING: Dict[Backend, Callable] = OrderedDict(
    [
        (Backend.FLAX, deserialize_flax),
        (Backend.NUMPY, deserialize_numpy),
        (Backend.PADDLEPADDLE, deserialize_paddle),
        # (Backend.TENSORFLOW, deserialize_tensorflow),
        (Backend.TORCH, deserialize_torch),
    ]
)
# Mapping between tensor type and backend
TENSOR_TYPE_MAPPING: Dict[
    Union[jaxlib.xla_extension.ArrayImpl, np.ndarray, paddle.Tensor, torch.Tensor],
    Backend,
] = OrderedDict(
    [
        (jaxlib.xla_extension.ArrayImpl, Backend.FLAX),
        (np.ndarray, Backend.NUMPY),
        (paddle.Tensor, Backend.PADDLEPADDLE),
        # (tf.Tensor, Backend.TENSORFLOW),
        (torch.Tensor, Backend.TORCH),
    ]
)
