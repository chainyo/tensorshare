"""Tensorflow serialization and deserialization utilities."""

from typing import Dict, Optional

import tensorflow as tf
from safetensors.tensorflow import load as tf_load
from safetensors.tensorflow import save as tf_save

from tensorshare.import_utils import require_backend


@require_backend("tensorflow")
def serialize(
    tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Convert tensorflow tensors to safetensors format using `safetensors.tensorflow.save`.

    Args:
        tensors (Dict[str, tf.Tensor]):
            Tensorflow tensors stored in a dictionary with their name as key.
        metadata (Optional[Dict[str, str]], optional):
            Metadata to add to the safetensors file. Defaults to None.

    Returns:
        bytes: Tensors formatted with their metadata if any.
    """
    return tf_save(tensors, metadata=metadata)  # type: ignore


@require_backend("tensorflow")
def deserialize(data: bytes) -> Dict[str, tf.Tensor]:
    """
    Convert safetensors format to tensorflow tensors using `safetensors.tensorflow.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, tf.Tensor]: Tensorflow tensors stored in a dictionary with their name as key.
    """
    return tf_load(data)  # type: ignore
