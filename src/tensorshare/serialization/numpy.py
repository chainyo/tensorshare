"""NumPy serialization and deserialization utilities."""

from typing import Dict, Optional

import numpy as np
from safetensors.numpy import load as np_load
from safetensors.numpy import save as np_save

from tensorshare.import_utils import require_backend


@require_backend("numpy")
def serialize(
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
    return np_save(tensors, metadata=metadata)  # type: ignore


@require_backend("numpy")
def deserialize(data: bytes) -> Dict[str, np.ndarray]:
    """
    Convert safetensors format to numpy tensors using `safetensors.numpy.load`.

    Args:
        data (bytes): Safetensors formatted data.

    Returns:
        Dict[str, np.ndarray]: Numpy tensors stored in a dictionary with their name as key.
    """
    return np_load(data)  # type: ignore
