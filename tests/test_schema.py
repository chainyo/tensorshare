"""Tests for the pydantic schemas of tensorshare."""

import jax.numpy as jnp
import numpy as np
import paddle
import pytest

# import tensorflow as tf
import torch
from pydantic import ByteSize

from tensorshare.schema import TensorShare
from tensorshare.serialization.constants import Backend
from tensorshare.serialization.utils import (
    serialize_flax,
    serialize_numpy,
    serialize_paddle,
    # serialize_tensorflow,
    serialize_torch,
)


class TestTensorShare:
    """Tests for the pydantic schemas of tensorshare."""

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    def test_tensorshare_pydantic_schema_with_fixed_tensor(
        self, serialized_fixed_numpy_tensors
    ) -> None:
        """Test the pydantic schema of tensorshare."""
        tensorshare = TensorShare(
            tensors=serialized_fixed_numpy_tensors,
            size=len(serialized_fixed_numpy_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert isinstance(tensorshare.size_as_str, str)
        assert tensorshare.size_as_str == "112B"

        assert (
            tensorshare.model_dump_json(indent=4)
            == '{\n    "tensors":'
            ' "SAAAAAAAAAB7ImVtYmVkZGluZ3MiOnsiZHR5cGUiOiJGNjQiLCJzaGFwZSI6WzIsMl0sImRhdGFfb2Zmc2V0cyI6WzAsMzJdfX0gICAgICAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==",\n'
            '    "size": 112\n}'
        )

    @pytest.mark.usefixtures("random_fp32_numpy_tensors")
    def test_tensorshare_pydantic_schema_with_random_tensor(
        self, random_fp32_numpy_tensors
    ) -> None:
        """Test the pydantic schema of tensorshare."""
        converted_tensors = serialize_numpy(random_fp32_numpy_tensors)

        tensorshare = TensorShare(
            tensors=converted_tensors,
            size=len(converted_tensors),
        )

        assert tensorshare.tensors == converted_tensors
        assert tensorshare.size == len(converted_tensors)

        assert (
            tensorshare.size_as_str == ByteSize(len(converted_tensors)).human_readable()
        )

    @pytest.mark.usefixtures("dict_zeros_flax_tensor")
    def test_tensorshare_with_flax(self, dict_zeros_flax_tensor) -> None:
        """Test the tensorshare schema with flax tensors."""
        _tensors = serialize_flax(dict_zeros_flax_tensor)
        tensorshare = TensorShare(
            tensors=_tensors,
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == _tensors
        assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_numpy_tensor")
    def test_tensorshare_with_numpy(self, dict_zeros_numpy_tensor) -> None:
        """Test the tensorshare schema with numpy tensors."""
        _tensors = serialize_numpy(dict_zeros_numpy_tensor)
        tensorshare = TensorShare(
            tensors=_tensors,
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == _tensors
        assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_paddle_tensor")
    def test_tensorshare_with_paddle(self, dict_zeros_paddle_tensor) -> None:
        """Test the tensorshare schema with paddle tensors."""
        _tensors = serialize_paddle(dict_zeros_paddle_tensor)
        tensorshare = TensorShare(
            tensors=_tensors,
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == _tensors
        assert tensorshare.size == len(_tensors)

        # @pytest.mark.usefixtures("dict_zeros_tensorflow_tensor")
        # def test_tensorshare_with_tensorflow(self, dict_zeros_tensorflow_tensor) -> None:
        """Test the tensorshare schema with tensorflow tensors."""
        # _tensors = serialize_tensorflow(dict_zeros_tensorflow_tensor)
        # tensorshare = TensorShare(
        #     tensors=_tensors,
        #     size=len(_tensors),
        # )

        # assert isinstance(tensorshare.tensors, bytes)
        # assert isinstance(tensorshare.size, int)

        # assert len(tensorshare.tensors) > 0
        # assert tensorshare.size > 0

        # assert tensorshare.tensors == _tensors
        # assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_torch_tensor")
    def test_tensorshare_with_torch(self, dict_zeros_torch_tensor) -> None:
        """Test the tensorshare schema with torch tensors."""
        _tensors = serialize_torch(dict_zeros_torch_tensor)
        tensorshare = TensorShare(
            tensors=_tensors,
            size=len(_tensors),
        )

        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == _tensors
        assert tensorshare.size == len(_tensors)

    @pytest.mark.usefixtures("dict_zeros_flax_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.FLAX, "flax"])
    def test_from_dict_method_with_flax(self, dict_zeros_flax_tensor, backend) -> None:
        """Test the from_dict method with flax tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_flax_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == serialize_flax(dict_zeros_flax_tensor)
        assert tensorshare.size == ByteSize(len(serialize_flax(dict_zeros_flax_tensor)))

    @pytest.mark.usefixtures("dict_zeros_numpy_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.NUMPY, "numpy"])
    def test_from_dict_method_with_numpy(
        self, dict_zeros_numpy_tensor, backend
    ) -> None:
        """Test the from_dict method with numpy tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_numpy_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

    @pytest.mark.usefixtures("dict_zeros_paddle_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_from_dict_method_with_paddle(
        self, dict_zeros_paddle_tensor, backend
    ) -> None:
        """Test the from_dict method with paddle tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_paddle_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

    # @pytest.mark.usefixtures("dict_zeros_tensorflow_tensor")
    # @parametrize("backend", [None, Backend.TENSORFLOW, "tensorflow"])
    # def test_from_dict_method_with_tensorflow(self, dict_zeros_tensorflow_tensor, backend) -> None:
    #     """Test the from_dict method with tensorflow tensors."""
    #     tensorshare = TensorShare.from_dict(dict_zeros_tensorflow_tensor, backend=backend)

    #     assert isinstance(tensorshare, TensorShare)
    #     assert isinstance(tensorshare.tensors, bytes)
    #     assert isinstance(tensorshare.size, int)

    #     assert len(tensorshare.tensors) > 0
    #     assert tensorshare.size > 0

    #     assert tensorshare.tensors == serialize_tensorflow(
    #         dict_zeros_tensorflow_tensor
    #     )
    #     assert tensorshare.size == len(
    #         serialize_tensorflow(dict_zeros_tensorflow_tensor)
    #     )

    @pytest.mark.usefixtures("dict_zeros_torch_tensor")
    @pytest.mark.parametrize("backend", [None, Backend.TORCH, "torch"])
    def test_from_dict_method_with_torch(
        self, dict_zeros_torch_tensor, backend
    ) -> None:
        """Test the from_dict method with torch tensors."""
        tensorshare = TensorShare.from_dict(dict_zeros_torch_tensor, backend=backend)

        assert isinstance(tensorshare, TensorShare)
        assert isinstance(tensorshare.tensors, bytes)
        assert isinstance(tensorshare.size, int)

        assert len(tensorshare.tensors) > 0
        assert tensorshare.size > 0

        assert tensorshare.tensors == serialize_torch(dict_zeros_torch_tensor)
        assert tensorshare.size == ByteSize(
            len(serialize_torch(dict_zeros_torch_tensor))
        )

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.FLAX, "flax"])
    def test_to_tensors_method_with_flax(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with flax backend."""
        tensorshare = TensorShare(
            tensors=serialized_fixed_numpy_tensors,
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], jnp.ndarray)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.NUMPY, "numpy"])
    def test_to_tensors_method_with_numpy(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with numpy backend."""
        tensorshare = TensorShare(
            tensors=serialized_fixed_numpy_tensors,
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], np.ndarray)
        assert tensors["embeddings"].shape == (2, 2)

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.PADDLEPADDLE, "paddlepaddle"])
    def test_to_tensors_method_with_paddle(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with paddle backend."""
        tensorshare = TensorShare(
            tensors=serialized_fixed_numpy_tensors,
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], paddle.Tensor)
        assert tensors["embeddings"].shape == [2, 2]

    # @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    # @pytest.mark.parametrize("backend", [Backend.TENSORFLOW, "tensorflow"])
    # def test_to_tensors_method_with_tensorflow(self, serialized_fixed_numpy_tensors, backend) -> None:
    #     """Test the to_tensors method with tensorflow backend."""
    #     tensorshare = TensorShare(
    #         tensors=serialized_fixed_numpy_tensors,
    #         size=len(serialized_fixed_numpy_tensors)
    #     )
    #     tensors = tensorshare.to_tensors(backend=backend)

    #     assert isinstance(tensors, dict)
    #     assert isinstance(tensors["embeddings"], tf.Tensor)
    #     assert tensors["embeddings"].shape == [2, 2]

    @pytest.mark.usefixtures("serialized_fixed_numpy_tensors")
    @pytest.mark.parametrize("backend", [Backend.TORCH, "torch"])
    def test_to_tensors_method_with_torch(
        self, serialized_fixed_numpy_tensors, backend
    ) -> None:
        """Test the to_tensors method with torch backend."""
        tensorshare = TensorShare(
            tensors=serialized_fixed_numpy_tensors,
            size=len(serialized_fixed_numpy_tensors),
        )
        tensors = tensorshare.to_tensors(backend=backend)

        assert isinstance(tensors, dict)
        assert isinstance(tensors["embeddings"], torch.Tensor)
        assert tensors["embeddings"].shape == torch.Size([2, 2])
