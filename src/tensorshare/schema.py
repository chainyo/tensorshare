"""Pydantic schemas to enable resilient and secure tensor sharing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import paddle
    import tensorflow as tf
    import torch
    from jax import Array

from pydantic import BaseModel, ByteSize, ConfigDict, Field, HttpUrl

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


class DefaultResponse(BaseModel):
    """Base model for tensor sharing response endpoint."""

    message: str


class TensorShareServer(BaseModel):
    """
    TensorShare server schema used by FastAPI or the TensorShareClient.

    Raises:
        ValueError: If the url is not a valid url.
    """

    url: HttpUrl
    ping: HttpUrl
    receive_tensor: HttpUrl
    response_model: Type[BaseModel] = Field(
        default=DefaultResponse,
        validate_default=True,
        exclude=True,
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "url": "http://localhost:8000",
            "ping": "http://localhost:8000/ping",
            "receive_tensor": "http://localhost:8000/receive_tensor",
        },
    )

    @classmethod
    def from_dict(
        cls, server_config: Dict[str, Union[str, Type[BaseModel]]]
    ) -> TensorShareServer:
        """
        Create a TensorShareServer object from a dictionary.

        Args:
            server_config (Dict[str, Union[str, Type[BaseModel]]]):
                Dictionary containing the server configuration.
                The expected keys are `url`, `ping`, `receive_tensor` and `response_model`.
                Only `url` is mandatory, `ping` and `receive_tensor` will be set to default values,
                and `response_model` will be set to DefaultResponse.

        Raises:
            ValueError: If the url is not a valid url or missing.

        Returns:
            TensorShareServer: A TensorShareServer object.
        """
        url = server_config.get("url", None)

        if url is None:
            raise ValueError("Verify the configuration dictionary, `url` is missing.")

        ping = HttpUrl(f"{url}/{server_config.get('ping', 'ping')}")
        receive_tensor = HttpUrl(
            f"{url}/{server_config.get('receive_tensor', 'receive_tensor')}"
        )
        response_model = server_config.get("response_model", DefaultResponse)

        return cls(
            url=url,
            ping=ping,
            receive_tensor=receive_tensor,
            response_model=response_model,
        )
