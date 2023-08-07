"""Tests for the pydantic schemas of tensorshare."""

import pytest
from pydantic import ByteSize

from tensorshare.converter.utils import (
    convert_flax_to_safetensors,
    convert_numpy_to_safetensors,
    convert_paddle_to_safetensors,
    # convert_tensorflow_to_safetensors,
    convert_torch_to_safetensors,
)
from tensorshare.schema import TensorShare


class TestTensorShare:
    """Tests for the pydantic schemas of tensorshare."""

    @pytest.mark.usefixtures("converted_fixed_numpy_tensors")
    def test_tensorshare_pydantic_schema_with_fixed_tensor(
        self, converted_fixed_numpy_tensors
    ) -> None:
        """Test the pydantic schema of tensorshare."""
        tensorshare = TensorShare(
            tensors=converted_fixed_numpy_tensors,
            size=len(converted_fixed_numpy_tensors),
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
        converted_tensors = convert_numpy_to_safetensors(random_fp32_numpy_tensors)

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
        _tensors = convert_flax_to_safetensors(dict_zeros_flax_tensor)
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
        _tensors = convert_numpy_to_safetensors(dict_zeros_numpy_tensor)
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
        _tensors = convert_paddle_to_safetensors(dict_zeros_paddle_tensor)
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
        # _tensors = convert_tensorflow_to_safetensors(dict_zeros_tensorflow_tensor)
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
        _tensors = convert_torch_to_safetensors(dict_zeros_torch_tensor)
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
