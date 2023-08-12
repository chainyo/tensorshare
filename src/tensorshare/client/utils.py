"""Utility functions for the Client module."""

from pydantic import ByteSize

from tensorshare.schema import TensorShare


def fake_tensorshare_data() -> TensorShare:
    """Returns a fake TensorShare object."""
    return TensorShare(
        tensors=(
            b'H\x00\x00\x00\x00\x00\x00\x00{"embeddings":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}'
            b"       \x00\x00\x00\x00"
        ),
        size=ByteSize(84),
    )
