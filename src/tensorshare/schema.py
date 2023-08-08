"""Pydantic schemas to enable resilient and secure tensor sharing."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import paddle

# import tensorflow as tf
import torch
from jax import Array
from pydantic import BaseModel, ByteSize, ConfigDict

from tensorshare.serialization.constants import Backend
from tensorshare.serialization.processor import TensorProcessor


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

    @staticmethod
    def from_dict(
        tensors: Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]],
        metadata: Optional[Dict[str, str]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ) -> TensorShare:
        """
        Create a TensorShare object from a dictionary of tensors.

        Args:
            tensors (Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]]):
                Tensors stored in a dictionary with their name as key.
            metadata (Optional[Dict[str, str]], optional):
                Metadata to add to the safetensors file. Defaults to None.
            backend (Optional[Union[str, Backend]], optional):
                Backend to use for the conversion. Defaults to None.

        Raises:
            TypeError: If tensors is not a dictionary.
            TypeError: If backend is not a string or an instance of Backend enum.
            ValueError: If backend is not supported.
            ValueError: If tensors is empty.

        Returns:
            TensorShare: A TensorShare object.
        """
        tensors, size = TensorProcessor.serialize(
            tensors=tensors, metadata=metadata, backend=backend
        )

        return TensorShare(tensors=tensors, size=size)
