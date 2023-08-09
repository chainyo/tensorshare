"""Define all the backends conversion methods to safetensors format."""

from typing import Dict, Optional

import numpy as np
import paddle

# import tensorflow as tf
import torch
from jax import Array
from safetensors.flax import load as flax_load
from safetensors.flax import save as flax_save
from safetensors.numpy import load as np_load
from safetensors.numpy import save as np_save
from safetensors.paddle import load as paddle_load
from safetensors.paddle import save as paddle_save
from safetensors.torch import load as torch_load

# from safetensors.tensorflow import save as tf_save, load as tf_load
from safetensors.torch import save as torch_save

from tensorshare.import_utils import require_backend


@require_backend("flax", "jax", "jaxlib")
def serialize_flax(
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


@require_backend("flax", "jax", "jaxlib")
def deserialize_flax(data: bytes) -> Dict[str, Array]:
    """
    Convert safetensors format to flax tensors using `safetensors.flax.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, Array]: Flax tensors stored in a dictionary with their name as key.
    """
    return flax_load(data)


@require_backend("numpy")
def serialize_numpy(
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


@require_backend("numpy")
def deserialize_numpy(data: bytes) -> Dict[str, np.ndarray]:
    """
    Convert safetensors format to numpy tensors using `safetensors.numpy.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, np.ndarray]: Numpy tensors stored in a dictionary with their name as key.
    """
    return np_load(data)


@require_backend("paddlepaddle")
def serialize_paddle(
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


@require_backend("paddlepaddle")
def deserialize_paddle(data: bytes) -> Dict[str, paddle.Tensor]:
    """
    Convert safetensors format to paddle tensors using `safetensors.paddle.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, paddle.Tensor]: Paddle tensors stored in a dictionary with their name as key.
    """
    return paddle_load(data)


# @require_backend("tensorflow")
# def serialize_tensorflow(
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


# @require_backend("tensorflow")
# def deserialize_tensorflow(data: bytes) -> Dict[str, tf.Tensor]:
#     """
#     Convert safetensors format to tensorflow tensors using `safetensors.tensorflow.load`.

#     Args:
#         data (bytes): Safetensors formatted data.

#     Returns:
#         Dict[str, tf.Tensor]: Tensorflow tensors stored in a dictionary with their name as key.
#     """
#     return tf_load(data)


@require_backend("torch")
def serialize_torch(
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


@require_backend("torch")
def deserialize_torch(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Convert safetensors format to torch tensors using `safetensors.torch.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, torch.Tensor]: Torch tensors stored in a dictionary with their name as key.
    """
    return torch_load(data)
