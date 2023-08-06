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
"""Regroup all the utilities for importing modules/libraries in a single file."""

import importlib
from functools import lru_cache
from typing import Callable


# This function is taken from the Hugging Face Transformers library:
# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py#L41
@lru_cache(maxsize=None)
def _is_package_available(package_name: str) -> bool:
    """
    Check if package is available.

    Args:
        package_name (str):
            Package name to check availability for.

    Returns:
        bool: Whether the package is available.
    """
    if importlib.util.find_spec(package_name) is not None:
        return True

    return False


@lru_cache(maxsize=None)
def _is_padddle_available() -> bool:
    """
    Check if paddle is available.

    Returns:
        bool: Whether the package is available.
    """
    try:
        import paddle  # noqa: F401

        package_available = True
    except ImportError:
        package_available = False
    finally:
        return package_available


def require_backend(*backend_names: str) -> Callable[[], Callable]:
    """
    A decorator that checks if the required backends are available.

    Args:
        *check_funcs (Callable): Functions that return True if the backend is available.

    Returns:
        Callable[[], Callable]: Decorator.
    """

    def decorator(func: Callable) -> Callable[[], Callable]:
        """Decorator."""

        def wrapper(*args, **kwargs) -> Callable:
            """Wrapper."""
            for backend_name in backend_names:
                if backend_name == "paddlepaddle":
                    if not _is_padddle_available():
                        raise ImportError(
                            f"`{func.__name__}` requires `paddle` to be installed."
                        )
                else:
                    if not _is_package_available(backend_name):
                        raise ImportError(
                            f"`{func.__name__}` requires `{backend_name}` to be"
                            " installed."
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator
