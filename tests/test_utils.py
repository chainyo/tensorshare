"""Test the utils functions of the Client module."""

from typing import Any, Dict

import pytest
from pydantic import ByteSize

from tensorshare.schema import TensorShare
from tensorshare.utils import fake_tensorshare_data, prepare_tensors_to_dict


class TestClientUtils:
    """Test the utils functions of the Client module."""

    def test_fake_tensorshare_data(self) -> None:
        """Test the fake_tensorshare_data function."""
        fake_ts = fake_tensorshare_data()

        assert isinstance(fake_ts, TensorShare)
        assert (
            fake_ts.tensors
            == b'H\x00\x00\x00\x00\x00\x00\x00{"embeddings":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}'
            b"       \x00\x00\x00\x00"
        )
        assert fake_ts.size == ByteSize(84)

    @pytest.mark.parametrize(
        "data, expected",
        [
            ({"embeddings": [1, 2, 3]}, {"embeddings": [1, 2, 3]}),
            (
                {"embeddings": [1, 2, 3], "labels": [1, 2, 3]},
                {"embeddings": [1, 2, 3], "labels": [1, 2, 3]},
            ),
            (
                [[1, 2, 3], [1, 2, 3]],
                {"embeddings_0": [1, 2, 3], "embeddings_1": [1, 2, 3]},
            ),
            (
                ([1, 2, 3], [1, 2, 3]),
                {"embeddings_0": [1, 2, 3], "embeddings_1": [1, 2, 3]},
            ),
            (
                {0: [1, 2, 3], 1: [1, 2, 3]},
                {"0": [1, 2, 3], "1": [1, 2, 3]},
            ),
            (
                (lambda: (i for i in [[1, 2, 3], [1, 2, 3], [1, 2, 3]])),
                {
                    "embeddings_0": [1, 2, 3],
                    "embeddings_1": [1, 2, 3],
                    "embeddings_2": [1, 2, 3],
                },
            ),
        ],
    )
    def test_prepare_tensors_to_dict(self, data: Any, expected: Dict[str, Any]) -> None:
        """Test the prepare_tensors_to_dict function."""
        data = data() if callable(data) else data

        assert prepare_tensors_to_dict(data) == expected
