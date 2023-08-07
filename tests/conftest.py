"""Fixture stores for the testing suite."""

import jax.numpy as jnp
import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch


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
