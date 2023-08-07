"""Pydantic schemas to enable resilient and secure tensor sharing."""

from pydantic import BaseModel, ByteSize, ConfigDict


class TensorShare(BaseModel):
    """Base model for tensor sharing."""

    tensors: bytes
    size: ByteSize

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "tensors": (
                b'H\x00\x00\x00\x00\x00\x00\x00{"embeddings":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}'
                b"       \x00\x00\x00\x00"
            ),
            "size": 84,
        },
        ser_json_bytes="base64",
    )

    @property
    def size_as_str(self) -> str:
        """Return size as a string in human readable format."""
        return self.size.human_readable()
