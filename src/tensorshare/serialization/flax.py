"""Flax serialization and deserialization utilities."""

from typing import Dict, Optional

from jax import Array
from safetensors.flax import load as flax_load
from safetensors.flax import save as flax_save

from tensorshare.import_utils import require_backend


@require_backend("flax", "flaxlib", "jax")
def serialize(
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
    return flax_save(tensors, metadata=metadata)  # type: ignore


@require_backend("flax", "flaxlib", "jax")
def deserialize(data: bytes) -> Dict[str, Array]:
    """
    Convert safetensors format to flax tensors using `safetensors.flax.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, Array]: Flax tensors stored in a dictionary with their name as key.
    """
    return flax_load(data)  # type: ignore
