"""PaddlePaddle serialization and deserialization utilities"""

from typing import Dict, Optional

import paddle
from safetensors.paddle import load as paddle_load
from safetensors.paddle import save as paddle_save

from tensorshare.import_utils import require_backend


@require_backend("paddle")
def serialize(
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
    return paddle_save(tensors, metadata=metadata)  # type: ignore


@require_backend("paddle")
def deserialize(data: bytes) -> Dict[str, paddle.Tensor]:
    """
    Convert safetensors format to paddle tensors using `safetensors.paddle.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, paddle.Tensor]: Paddle tensors stored in a dictionary with their name as key.
    """
    return paddle_load(data)  # type: ignore
