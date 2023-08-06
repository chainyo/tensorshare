# Copyright 2023 Thomas Chaigneau. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils functions for the converter module."""

from typing import Dict, Optional

import numpy as np
import paddle
# import tensorflow as tf
import torch
from jax import Array
from safetensors.flax import save as flax_save
from safetensors.numpy import save as np_save
from safetensors.paddle import save as paddle_save
# from safetensors.tensorflow import save as tf_save
from safetensors.torch import save as torch_save


def convert_flax_to_safetensors(
    tensors: Dict[str, Array], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Convert flax tensors to safetensors format using `safetensors.flax.save`.

    Args:
        tensors (Dict[str, Array]):
            Flax tensors stored in a dictionary with their name as key.
        metadata (Optional[Dict[str, str]], optional):
            Metadata to add to the safetensors file. Defaults to None.

    Returns:
        bytes: Tensors formatted with their metadata if any.
    """
    return flax_save(tensors, metadata=metadata)

def convert_numpy_to_safetensors(
    tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Convert numpy tensors to safetensors format using `safetensors.numpy.save`.

    Args:
        tensors (Dict[str, np.ndarray]):
            Numpy tensors stored in a dictionary with their name as key.
        metadata (Optional[Dict[str, str]], optional):
            Metadata to add to the safetensors file. Defaults to None.

    Returns:
        bytes: Tensors formatted with their metadata if any.
    """
    return np_save(tensors, metadata=metadata)

def convert_paddle_to_safetensors(
    tensors: Dict[str, paddle.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Convert paddle tensors to safetensors format using `safetensors.paddle.save`.

    Args:
        tensors (Dict[str, paddle.Tensor]):
            Paddle tensors stored in a dictionary with their name as key.
        metadata (Optional[Dict[str, str]], optional):
            Metadata to add to the safetensors file. Defaults to None.

    Returns:
        bytes: Tensors formatted with their metadata if any.
    """
    return paddle_save(tensors, metadata=metadata)

# def convert_tensorflow_to_safetensors(
#     tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None
# ) -> bytes:
#     """
#     Convert tensorflow tensors to safetensors format using `safetensors.tensorflow.save`.

#     Args:
#         tensors (Dict[str, tf.Tensor]):
#             Tensorflow tensors stored in a dictionary with their name as key.
#         metadata (Optional[Dict[str, str]], optional):
#             Metadata to add to the safetensors file. Defaults to None.

#     Returns:
#         bytes: Tensors formatted with their metadata if any.
#     """
#     return tf_save(tensors, metadata=metadata)

def convert_torch_to_safetensors(
    tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Convert torch tensors to safetensors format using `safetensors.torch.save`.

    Args:
        tensors (Dict[str, torch.Tensor]):
            Torch tensors stored in a dictionary with their name as key.
        metadata (Optional[Dict[str, str]], optional):
            Metadata to add to the safetensors file. Defaults to None.

    Returns:
        bytes: Tensors formatted with their metadata if any.
    """
    return torch_save(tensors, metadata=metadata)
