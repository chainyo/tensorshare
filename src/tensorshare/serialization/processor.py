"""Processor interface for converting between different tensor formats."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import paddle

# import tensorflow as tf
import torch
from jax import Array
from pydantic import ByteSize

from tensorshare.serialization.constants import (
    BACKENDS_FUNC_MAPPING,
    TENSOR_TYPE_MAPPING,
    Backend,
)


def _infer_backend(
    tensors: Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]],
) -> Backend:
    """Find the backend of the tensors based on their type.

    This method will try to infer the backend by checking the type of the first tensor in the dictionary
    and comparing it to the supported types. If the type is not supported, it will raise a ValueError.
    It will also check if all the tensors have the same type if `check_all_same_type` is True (default).

    Args:
        tensors (Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]]):
            Tensors stored in a dictionary with their name as key.

    Raises:
        TypeError: If all tensors don't have the same type.
        TypeError: If the type of the tensors is not supported.

    Returns:
        Backend: The backend inferred from the type of the tensors.
    """
    tensor_types = [type(tensor) for tensor in tensors.values()]

    if len(set(tensor_types)) > 1:
        raise TypeError(
            f"All tensors must have the same type, got {tensor_types} instead."
        )

    first_tensor_type = tensor_types[0]

    if first_tensor_type not in TENSOR_TYPE_MAPPING:
        raise TypeError(
            f"Unsupported tensor type {first_tensor_type}. Supported types are"
            f" {list(TENSOR_TYPE_MAPPING.keys())}\nThe supported backends are"
            f" {list(BACKENDS_FUNC_MAPPING.keys())}."
        )

    return TENSOR_TYPE_MAPPING[first_tensor_type]


class TensorProcessor:
    """Tensor processor class."""

    @staticmethod
    def serialize(
        tensors: Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]],
        metadata: Optional[Dict[str, str]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ) -> Tuple[bytes, ByteSize]:
        """Serialize a dictionary of tensors to a TensorShare object.

        This method will convert a dictionary of tensors to a TensorShare object using the specified backend
        if provided, otherwise it will try to infer the backend from the tensors format.

        Args:
            tensors (Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]]):
                Tensors stored in a dictionary with their name as key.
            metadata (Optional[Dict[str, str]], optional):
                Metadata to add to the safetensors file. Defaults to None.
            backend (Optional[Union[str, Backend]], optional):
                Backend to use for the conversion. Defaults to None.
                If None, the backend will be inferred from the tensors format.
                Backend can be one of the following:
                    - Backend.FLAX or 'flax'
                    - Backend.NUMPY or 'numpy'
                    - Backend.PADDLEPADDLE or 'paddlepaddle'
                    - Backend.TENSORFLOW or 'tensorflow'
                    - Backend.TORCH or 'torch'

        Raises:
            TypeError: If tensors is not a dictionary.
            TypeError: If backend is not a string or an instance of Backend enum.
            ValueError: If tensors is empty.
            KeyError: If backend is not one of the supported backends.

        Returns:
            Tuple[bytes, ByteSize]: A tuple containing the serialized tensors and the size of the file.
        """
        if not isinstance(tensors, dict):
            raise TypeError(
                f"Tensors must be a dictionary, got `{type(tensors)}` instead."
            )
        elif not tensors:
            raise ValueError("Tensors dictionary cannot be empty.")

        if backend is not None:
            if isinstance(backend, str):
                try:
                    _backend = Backend[backend.upper()]
                except KeyError as e:
                    raise KeyError(
                        f"Invalid backend `{backend}`. Must be one of"
                        f" {list(Backend.__members__)}."
                    ) from e
            elif not isinstance(backend, Backend):
                raise TypeError(
                    "Backend must be a string or an instance of Backend enum, got"
                    f" `{type(backend)}` instead. Use `tensorshare.schema.Backend` to"
                    " access the Backend enum. If you don't specify a backend, it will"
                    " be inferred from the tensors format."
                )
        else:
            _backend = _infer_backend(tensors)

        _tensors = BACKENDS_FUNC_MAPPING[_backend](tensors, metadata=metadata)

        return _tensors, ByteSize(len(_tensors))

    @staticmethod
    def deserialize() -> None:
        """"""
        pass
