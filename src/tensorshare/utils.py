"""Utility functions for the Client module."""

import types
from typing import Any, Dict

from pydantic import ByteSize

from tensorshare.schema import TensorShare


def fake_tensorshare_data() -> TensorShare:
    """Get a fake TensorShare object.

    Returns a fake TensorShare object with non base64 encoded data. It is
    only for testing purposes or the TensorShareClient initialization.
    The usual tensors value will be pre-encoded to base64.

    Returns:
        TensorShare: A fake TensorShare object.
    """
    return TensorShare(
        tensors=(
            b'H\x00\x00\x00\x00\x00\x00\x00{"embeddings":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}'
            b"       \x00\x00\x00\x00"
        ),
        size=ByteSize(84),
    )


def prepare_tensors_to_dict(data: Any) -> Dict[str, Any]:
    """Prepare the tensors from any framework to a dictionary.

    This function is used to prepare the tensors from any framework to a
    dictionary. The dictionary will be used to create a TensorShare object.
    Use this function as a lazy way to prepare the tensors.
    If the data is a dictionary, the keys will be used as the tensor names.
    If the data is a list, a set, a tuple or any other iterable, the tensors
    will be named "embeddings_{i}" where i is the index of the tensor in the
    iterable.

    Args:
        data (Any):
            The tensors to prepare. It can be a dictionary, a generator,
            a list, a set, a tuple or any other single item.

    Returns:
        Dict[str, Any]: The prepared data.
    """
    if isinstance(data, dict):
        return {str(key): tensor for key, tensor in data.items()}

    elif isinstance(data, (types.GeneratorType, list, set, tuple)) or hasattr(
        data, "__iter__"
    ):
        # The `hasattr(data, "__iter__")` check ensures that even if the input is some other kind of iterable
        # (not explicitly listed), it will still be processed here.
        return {f"embeddings_{i}": tensor for i, tensor in enumerate(data)}

    else:
        return {"embeddings": data}
