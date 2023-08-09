"""Torch serialization and deserialization utilities."""

from typing import Dict, Optional

import torch
from safetensors.torch import load as torch_load
from safetensors.torch import save as torch_save

from tensorshare.import_utils import require_backend


@require_backend("torch")
def serialize(
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
    return torch_save(tensors, metadata=metadata)  # type: ignore


@require_backend("torch")
def deserialize(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Convert safetensors format to torch tensors using `safetensors.torch.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, torch.Tensor]: Torch tensors stored in a dictionary with their name as key.
    """
    return torch_load(data)  # type: ignore
