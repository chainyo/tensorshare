"""Converter interface for converting between different tensor formats."""

from typing import Dict, Optional, OrderedDict, Union

import jaxlib
import numpy as np
import paddle

# import tensorflow as tf
import torch
from jax import Array
from pydantic import ByteSize

from tensorshare.converter.utils import (
    convert_flax_to_safetensors,
    convert_numpy_to_safetensors,
    convert_paddle_to_safetensors,
    # convert_tensorflow_to_safetensors,
    convert_torch_to_safetensors,
)
from tensorshare.schema import Backend, TensorShare

# Mapping between backend and conversion function
BACKENDS_FUNC_MAPPING = OrderedDict(
    [
        (Backend.FLAX, convert_flax_to_safetensors),
        (Backend.NUMPY, convert_numpy_to_safetensors),
        (Backend.PADDLEPADDLE, convert_paddle_to_safetensors),
        # (Backend.TENSORFLOW, convert_tensorflow_to_safetensors),
        (Backend.TORCH, convert_torch_to_safetensors),
    ]
)
# Mapping between tensor type and backend
TENSOR_TYPE_MAPPING = OrderedDict(
    [
        (jaxlib.xla_extension.ArrayImpl, Backend.FLAX),
        (np.ndarray, Backend.NUMPY),
        (paddle.Tensor, Backend.PADDLEPADDLE),
        # (tf.Tensor, Backend.TENSORFLOW),
        (torch.Tensor, Backend.TORCH),
    ]
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


class TensorConverter:
    """Tensor converter class."""

    @staticmethod
    def convert(
        tensors: Dict[str, Union[Array, np.ndarray, paddle.Tensor, torch.Tensor]],
        metadata: Optional[Dict[str, str]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ) -> TensorShare:
        """Convert a dictionary of tensors to a TensorShare object.

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
            TensorShare: TensorShare object containing the converted tensors.
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

        return TensorShare(
            tensors=_tensors,
            size=ByteSize(len(_tensors)),
        )
