# Copyright 2023 Thomas Chaigneau. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utils functions for the converter module."""

import pytest

from tensorshare.converter.utils import (
    convert_flax_to_safetensors,
    convert_numpy_to_safetensors,
    convert_paddle_to_safetensors,
    # convert_tensorflow_to_safetensors,
    convert_torch_to_safetensors,
)


class TestConvertionUtilsFunction:
    """Test utils functions for the converter module."""

    @pytest.mark.usefixtures("zeros_flax_tensor")
    def test_convert_flax_to_safetensors(self, zeros_flax_tensor) -> None:
        """Test convertion from flax to safetensors."""
        _tensors = {"embeddings": zeros_flax_tensor}
        tensor_bytes = convert_flax_to_safetensors(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_numpy_tensor")
    def test_convert_numpy_to_safetensors(self, zeros_numpy_tensor) -> None:
        """Test convertion from numpy to safetensors."""
        _tensors = {"embeddings": zeros_numpy_tensor}
        tensor_bytes = convert_numpy_to_safetensors(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_paddle_tensor")
    def test_convert_paddle_to_safetensors(self, zeros_paddle_tensor) -> None:
        """Test convertion from paddle to safetensors."""
        _tensors = {"embeddings": zeros_paddle_tensor}
        tensor_bytes = convert_paddle_to_safetensors(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0

    # @pytest.mark.usefixtures("zeros_tensorflow_tensor")
    # def test_convert_tensorflow_to_safetensors(self, zeros_tensorflow_tensor) -> None:
    #     """Test convertion from tensorflow to safetensors."""
    #     tensor_bytes = convert_tensorflow_to_safetensors(zeros_tensorflow_tensor)
    #
    #     assert isinstance(tensor_bytes, bytes)
    #     assert len(tensor_bytes) > 0

    @pytest.mark.usefixtures("zeros_torch_tensor")
    def test_convert_torch_to_safetensors(self, zeros_torch_tensor) -> None:
        """Test convertion from torch to safetensors."""
        _tensors = {"embeddings": zeros_torch_tensor}
        tensor_bytes = convert_torch_to_safetensors(_tensors)

        assert isinstance(tensor_bytes, bytes)
        assert len(tensor_bytes) > 0
