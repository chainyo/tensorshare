"""Fixture stores for the testing suite."""

import random

import jax.numpy as jnp
import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch
from aioresponses import aioresponses

from tensorshare.serialization.numpy import serialize


@pytest.fixture
def mock_server():
    with aioresponses() as m:
        yield m


@pytest.fixture
def serialized_fixed_numpy_tensors() -> bytes:
    """Return a serialized numpy tensor."""
    _tensor = {"embeddings": np.zeros((2, 2))}
    return serialize(_tensor)


@pytest.fixture
def random_fp32_numpy_tensors() -> bytes:
    """Return a serialized numpy tensor."""
    shape = (random.randint(1, 100), random.randint(1, 100))
    return {"embeddings": np.random.random(shape)}


@pytest.fixture
def multiple_backend_tensors() -> dict:
    """Return a dictionary of tensors from multiple backends."""
    return {
        "numpy": np.zeros((512, 1024)),
        "paddle": paddle.zeros((512, 1024)),
        "torch": torch.zeros((512, 1024)),
    }


@pytest.fixture
def dict_zeros_flax_tensor() -> dict:
    """Return a dictionary of flax tensors."""
    return {"embeddings": jnp.zeros((512, 1024))}


@pytest.fixture
def dict_zeros_numpy_tensor() -> dict:
    """Return a dictionary of numpy tensors."""
    return {"embeddings": np.zeros((512, 1024))}


@pytest.fixture
def dict_zeros_paddle_tensor() -> dict:
    """Return a dictionary of paddle tensors."""
    return {"embeddings": paddle.zeros((512, 1024))}


# @pytest.fixture
# def dict_zeros_tensorflow_tensor() -> dict:
#     """Return a dictionary of tensorflow tensors."""
#     return {"embeddings": tf.zeros((512, 1024))}


@pytest.fixture
def dict_zeros_torch_tensor() -> dict:
    """Return a dictionary of torch tensors."""
    return {"embeddings": torch.zeros((512, 1024))}


@pytest.fixture
def zeros_flax_tensor() -> jnp.ndarray:
    """Return a flax tensor of zeros."""
    return jnp.zeros((512, 1024))


@pytest.fixture
def zeros_numpy_tensor() -> np.ndarray:
    """Return a numpy tensor of zeros."""
    return np.zeros((512, 1024))


@pytest.fixture
def zeros_paddle_tensor() -> paddle.Tensor:
    """Return a paddle tensor of zeros."""
    return paddle.zeros((512, 1024))


# @pytest.fixture
# def zeros_tensorflow_tensor() -> tf.Tensor:
#     """Return a tensorflow tensor of zeros."""
#     return tf.zeros((512, 1024))


@pytest.fixture
def zeros_torch_tensor() -> torch.Tensor:
    """Return a torch tensor of zeros."""
    return torch.zeros((512, 1024))


@pytest.fixture
def multiple_flax_tensors() -> dict:
    """Return a dictionary of flax tensors."""
    return {
        "embeddings": jnp.zeros((512, 1024)),
        "labels": jnp.zeros((512, 1024)),
    }


@pytest.fixture
def multiple_numpy_tensors() -> dict:
    """Return a dictionary of numpy tensors."""
    return {
        "embeddings": np.zeros((512, 1024)),
        "labels": np.zeros((512, 1024)),
    }


@pytest.fixture
def multiple_paddle_tensors() -> dict:
    """Return a dictionary of paddle tensors."""
    return {
        "embeddings": paddle.zeros((512, 1024)),
        "labels": paddle.zeros((512, 1024)),
    }


# @pytest.fixture
# def multiple_tensorflow_tensors() -> dict:
#     """Return a dictionary of tensorflow tensors."""
#     return {
#         "embeddings": tf.zeros((512, 1024)),
#         "labels": tf.zeros((512, 1024)),
#     }


@pytest.fixture
def multiple_torch_tensors() -> dict:
    """Return a dictionary of torch tensors."""
    return {
        "embeddings": torch.zeros((512, 1024)),
        "labels": torch.zeros((512, 1024)),
    }
