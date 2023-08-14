"""Test the utils functions of the Client module."""

from pydantic import ByteSize

from tensorshare.schema import TensorShare
from tensorshare.utils import fake_tensorshare_data


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
