"""Constants for tensorshare."""

from enum import Enum
from typing import OrderedDict

import jaxlib
import numpy as np
import paddle

# import tensorflow as tf
import torch

from tensorshare.serialization.utils import (
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


# Mapping between backend and conversion function
BACKENDS_FUNC_MAPPING = OrderedDict(
    [
        (Backend.FLAX, serialize_flax),
        (Backend.NUMPY, serialize_numpy),
        (Backend.PADDLEPADDLE, serialize_paddle),
        # (Backend.TENSORFLOW, serialize_tensorflow),
        (Backend.TORCH, serialize_torch),
    ]
)
# Mapping between tensor type and backend
TENSOR_TYPE_MAPPING = OrderedDict(
    [
        (jaxlib.xla_extension.ArrayImpl, Backend.FLAX),
        (np.ndarray, Backend.NUMPY),
        (paddle.Tensor, Backend.PADDLEPADDLE),
        # (tf.Tensor, Backend.TENSORFLOW),
        (torch.Tensor, Backend.TORCH),
    ]
)
