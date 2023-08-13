"""Processor interface for converting between different tensor formats."""

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import paddle
    import tensorflow as tf
    import torch
    from jax import Array

from pydantic import ByteSize

from tensorshare.serialization.constants import (
    BACKEND_MODULE_MAPPING,
    BACKEND_TENSOR_TYPE_MAPPING,
    Backend,
    TensorType,
)


def _get_backend_method(backend: Backend, func_name: str) -> Any:
    """
    Get the function from the backend module.

    Args:
        backend: The backend to get the function for.
        func_name: The name of the function to get.

    Returns:
        The function from the backend module.
    """
    module_name = BACKEND_MODULE_MAPPING[backend]
    module = __import__(module_name, fromlist=[""])

    return getattr(module, func_name)


def _is_base64_encoded(data: Any) -> bool:
    # Regular expression to match base64 encoded string pattern
    try:
        decoded_data = base64.b64decode(data)
        encoded_data = base64.b64encode(decoded_data)
        return encoded_data == data
    except:
        return False


def _infer_backend(
    tensors: Dict[
        str, Union["Array", "np.ndarray", "paddle.Tensor", "tf.Tensor", "torch.Tensor"]
    ],
) -> Backend:
    """Find the backend of the tensors based on their type.

    This method will try to infer the backend by checking the type of the first tensor in the dictionary
    and comparing it to the supported types. If the type is not supported, it will raise a ValueError.
    It will also check if all the tensors have the same type if `check_all_same_type` is True (default).

    Args:
        tensors (Dict[str, Union[Array, np.ndarray, paddle.Tensor, tf.Tensor, torch.Tensor]]):
            Tensors stored in a dictionary with their name as key.

    Raises:
        TypeError: If all tensors don't have the same type.
        TypeError: If the type of the tensors is not supported.

    Returns:
        Backend: The backend inferred from the type of the tensors.
    """
    tensor_types = [
        f"{type(tensor).__module__}.{type(tensor).__name__}"
        for tensor in tensors.values()
    ]

    if len(set(tensor_types)) > 1:
        raise TypeError(
            f"All tensors must have the same type, got {tensor_types} instead."
        )

    first_tensor_type = tensor_types[0]

    try:
        backend = BACKEND_TENSOR_TYPE_MAPPING[first_tensor_type]  # type: ignore

    except KeyError as e:
        raise TypeError(
            f"Unsupported tensor type {first_tensor_type}. Supported types are"
            f" {list(TensorType)}\nThe supported backends are"
            f" {list(Backend)}."
        ) from e

    return backend


class TensorProcessor:
    """Tensor processor class."""

    @staticmethod
    def serialize(
        tensors: Dict[
            str,
            Union["Array", "np.ndarray", "paddle.Tensor", "tf.Tensor", "torch.Tensor"],
        ],
        metadata: Optional[Dict[str, str]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ) -> Tuple[bytes, ByteSize]:
        """Serialize a dictionary of tensors to a tuple containing the serialized tensors and their size.

        This method will convert a dictionary of tensors to a tuple containing the base64 encoded serialized
        tensors and the size of the serialized tensors. It will use the backend if provided, otherwise it will
        try to infer the backend from the tensors format.

        Args:
            tensors (Dict[str, Union[Array, np.ndarray, paddle.Tensor, tf.Tensor, torch.Tensor]]):
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
            Tuple[bytes, ByteSize]:
                A tuple containing the base64 encoded serialized tensors and the size of the serialized tensors.
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
                        f"Invalid backend `{backend}`. Must be one of {list(Backend)}."
                    ) from e

            elif not isinstance(backend, Backend):
                raise TypeError(
                    "Backend must be a string or an instance of Backend enum, got"
                    f" `{type(backend)}` instead. Use"
                    " `tensorshare.serialization.Backend` to access the Backend enum."
                    " If you don't specify a backend, it will be inferred from the"
                    " tensors format."
                )
        else:
            _backend = _infer_backend(tensors)

        _tensors = _get_backend_method(_backend, "serialize")(
            tensors, metadata=metadata
        )

        return base64.b64encode(_tensors), ByteSize(len(_tensors))

    @staticmethod
    def deserialize(
        data: bytes,
        backend: Union[str, Backend],
    ) -> Dict[
        str, Union["Array", "np.ndarray", "paddle.Tensor", "tf.Tensor", "torch.Tensor"]
    ]:
        """Deserialize base64 encoded serialized tensors to a dictionary of tensors.

        This method will convert TensorShare.tensors to a dictionary of tensors with their name as key.
        The backend must be specified in order to deserialize the data.

        Args:
            data (bytes):
                The base64 encoded serialized tensors to deserialize.
            backend (Union[str, Backend]):
                The backend to use for the conversion. Must be one of the following:
                    - Backend.FLAX or 'flax'
                    - Backend.NUMPY or 'numpy'
                    - Backend.PADDLEPADDLE or 'paddlepaddle'
                    - Backend.TENSORFLOW or 'tensorflow'
                    - Backend.TORCH or 'torch'

        Raises:
            TypeError: If data is not bytes.
            TypeError: If backend is not a string or an instance of Backend enum.
            KeyError: If backend is not one of the supported backends.

        Returns:
            Dict[str, Union[Array, np.ndarray, paddle.Tensor, tf.Tensor, torch.Tensor]]:
                A dictionary of tensors in the specified backend with their name as key.
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Data must be bytes, got `{type(data)}` instead.")

        if isinstance(backend, str):
            try:
                _backend = Backend[backend.upper()]
            except KeyError as e:
                raise KeyError(
                    f"Invalid backend `{backend}`. Must be one of {list(Backend)}."
                ) from e

        elif not isinstance(backend, Backend):
            raise TypeError(
                "Backend must be a string or an instance of Backend enum, got"
                f" `{type(backend)}` instead. Use `tensorshare.serialization.Backend`"
                " to access the Backend enum."
            )

        if _is_base64_encoded(data):
            data = base64.b64decode(data)

        tensors = _get_backend_method(_backend, "deserialize")(data)

        return tensors  # type: ignore
