"""Type hints for tensorshare.serialization.utils.py."""

from typing import Any, Dict, Optional

# Save functions from safetensors
def flax_save(tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]) -> bytes: ...
def np_save(tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]) -> bytes: ...
def paddle_save(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def tf_save(tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]) -> bytes: ...
def torch_save(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...

# Convert functions
def serialize_flax(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def serialize_numpy(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def serialize_paddle(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def serialize_tensorflow(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...
def serialize_torch(
    tensors: Dict[str, Any], metadata: Optional[Dict[str, str]]
) -> bytes: ...
