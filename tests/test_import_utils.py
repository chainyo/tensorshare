"""Tests for the import_utils functions."""

import pytest

from tensorshare.import_utils import (
    _is_flaxlib_available,
    _is_package_available,
    _is_paddle_available,
    require_backend,
)


@require_backend("torch")
def mock_function() -> None:
    """Mock function for testing the require_backend decorator."""
    pass


@require_backend("mxnet")
def mock_function_with_invalid_backend() -> None:
    """Mock function for testing the require_backend decorator."""
    pass


class TestImportUtils:
    """Tests for the import_utils functions."""

    def test_is_package_available(self) -> None:
        """Test the _is_package_available function."""
        for package in ["flax", "jax", "jaxlib", "numpy", "safetensors", "torch"]:
            assert _is_package_available(package) is True

    def test_is_package_available_with_cache(self) -> None:
        """Test the _is_package_available function with cache."""
        cache_info_before = _is_package_available.cache_info()
        _is_package_available("numpy")
        cache_info_after = _is_package_available.cache_info()

        assert cache_info_after.hits == cache_info_before.hits + 1
        assert cache_info_after.misses == cache_info_before.misses
        assert cache_info_after.maxsize == cache_info_before.maxsize
        assert cache_info_after.currsize == cache_info_before.currsize

    def test_non_available_package(self) -> None:
        """Test that _is_package_available returns False when a non-available package is passed.
        """
        for package in ["mxnet", "pandas", "werkzeug"]:
            assert _is_package_available(package) is False

    def test_invalid_package_name(self) -> None:
        """Test that _is_package_available raises an error when an invalid package name is passed.
        """
        with pytest.raises(ValueError):
            _is_package_available("")

    def test_is_flaxlib_available(self) -> None:
        """Test the _is_flaxlib_available function."""
        assert _is_flaxlib_available() is True

    def test_is_paddle_available(self) -> None:
        """Test the _is_paddle_available function."""
        assert _is_paddle_available() is True

    def test_is_paddle_available_with_cache(self) -> None:
        """Test the _is_paddle_available function with cache."""
        cache_info_before = _is_paddle_available.cache_info()
        _is_paddle_available()
        cache_info_after = _is_paddle_available.cache_info()

        assert cache_info_after.hits == cache_info_before.hits + 1
        assert cache_info_after.misses == cache_info_before.misses
        assert cache_info_after.maxsize == cache_info_before.maxsize
        assert cache_info_after.currsize == cache_info_before.currsize

    def test_require_backend(self) -> None:
        """Test the require_backend decorator."""
        mock_function()

    def test_require_backend_with_invalid_backend(self) -> None:
        """Test the require_backend decorator with an invalid backend."""
        with pytest.raises(ImportError) as excinfo:
            mock_function_with_invalid_backend()
            assert (
                "`mock_function_with_invalid_backend` requires `mxnet` to be installed."
                in str(excinfo.value)
            )
