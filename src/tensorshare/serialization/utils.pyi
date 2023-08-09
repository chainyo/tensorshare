"""Type hints for tensorshare.serialization.utils.py."""

from typing import Dict, Optional

import numpy as np
import paddle

# import tensorflow as tf
import torch
from jax import Array

# Load functions from safetensors
def flax_load(data: bytes) -> Dict[str, Array]: ...
def np_load(data: bytes) -> Dict[str, np.ndarray]: ...
def paddle_load(data: bytes) -> Dict[str, paddle.Tensor]: ...

# def tf_load(data: bytes) -> Dict[str, tf.Tensor]: ...
def torch_load(data: bytes) -> Dict[str, torch.Tensor]: ...

# Save functions from safetensors
def flax_save(
    tensors: Dict[str, Array], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def np_save(
    tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def paddle_save(
    tensors: Dict[str, paddle.Tensor], metadata: Optional[Dict[str, str]]
) -> bytes: ...

# def tf_save(tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]]) -> bytes: ...
def torch_save(
    tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]]
) -> bytes: ...

# Serialize functions
def serialize_flax(
    tensors: Dict[str, Array], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def serialize_numpy(
    tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def serialize_paddle(
    tensors: Dict[str, paddle.Tensor], metadata: Optional[Dict[str, str]]
) -> bytes: ...

# def serialize_tensorflow(
#     tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]]
# ) -> bytes: ...
def serialize_torch(
    tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]]
) -> bytes: ...

# Deserialize functions
def deserialize_flax(data: bytes) -> Dict[str, Array]: ...
def deserialize_numpy(data: bytes) -> Dict[str, np.ndarray]: ...
def deserialize_paddle(data: bytes) -> Dict[str, paddle.Tensor]: ...

# def deserialize_tensorflow(data: bytes) -> Dict[str, tf.Tensor]: ...
def deserialize_torch(data: bytes) -> Dict[str, torch.Tensor]: ...