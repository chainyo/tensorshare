"""Pydantic schemas to enable resilient and secure tensor sharing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import paddle
    import tensorflow as tf
    import torch
    from jax import Array

from pydantic import BaseModel, ByteSize, ConfigDict

from tensorshare.serialization import Backend, TensorProcessor


class TensorShare(BaseModel):
    """Base model for tensor sharing."""

    tensors: bytes
    size: ByteSize

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "tensors": b"SAAAAAAAAAB7ImVtYmVkZGluZ3MiOnsiZHR5cGUiOiJGMzIiLCJzaGFwZSI6WzEsMV0sImRhdGFfb2Zmc2V0cyI6WzAsNF19fSAgICAgICAAAAAA",
            "size": 84,
        },
    )

    @property
    def size_as_str(self) -> str:
        """Return size as a string in human readable format."""
        return self.size.human_readable()

    @classmethod
    def from_dict(
        cls,
        tensors: Dict[
            str,
            Union["Array", "np.ndarray", "paddle.Tensor", "tf.Tensor", "torch.Tensor"],
        ],
        metadata: Optional[Dict[str, str]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ) -> TensorShare:
        """
        Create a TensorShare object from a dictionary of tensors.

        Args:
            tensors (Dict[str, Union[Array, np.ndarray, paddle.Tensor, tf.Tensor, torch.Tensor]]):
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
        _tensors, size = TensorProcessor.serialize(
            tensors=tensors, metadata=metadata, backend=backend
        )

        return cls(tensors=_tensors, size=size)

    def to_tensors(
        self,
        backend: Union[str, Backend],
    ) -> Dict[
        str, Union["Array", "np.ndarray", "paddle.Tensor", "tf.Tensor", "torch.Tensor"]
    ]:
        """
        Convert a TensorShare object to a dictionary of tensors in the specified backend.

        Args:
            backend (Union[str, Backend]): Backend to use for the conversion.

        Raises:
            TypeError: If backend is not a string or an instance of Backend enum.
            ValueError: If backend is not supported.

        Returns:
            Dict[str, Union[Array, np.ndarray, paddle.Tensor, tf.Tensor, torch.Tensor]]:
                Tensors stored in a dictionary with their name as key in the specified backend.
        """
        return TensorProcessor.deserialize(self.tensors, backend=backend)
