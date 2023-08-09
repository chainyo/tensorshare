"""Regroup all the utilities for importing modules/libraries in a single file."""

import os
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from itertools import chain
from types import ModuleType
from typing import Any, Callable, Iterable, Optional, TypeVar

R = TypeVar("R")


# This function is inspired from the Hugging Face Transformers library:
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
    if find_spec(package_name) is not None:
        return True

    return False


@lru_cache(maxsize=None)
def _is_flaxlib_available() -> bool:  # pragma: no cover
    """
    Check if flaxlib is available.

    Returns:
        bool: Whether the package is available.
    """
    try:
        import jaxlib  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=None)
def _is_paddle_available() -> bool:  # pragma: no cover
    """
    Check if paddle is available.

    Returns:
        bool: Whether the package is available.
    """
    try:
        import paddle  # noqa: F401

        return True
    except ImportError:
        return False


def require_backend(
    *backend_names: str,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    A decorator that checks if the required backends are available.

    Args:
        *backend_names (str): Backend names to check availability for.

    Returns:
        Callable[[Callable[..., R]], Callable[..., R]]: Decorator.
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        """Decorator."""

        def wrapper(*args: Any, **kwargs: Any) -> R:
            """Wrapper."""
            for backend_name in backend_names:
                if backend_name == "paddlepaddle":
                    if not _is_paddle_available():
                        raise ImportError(
                            f"`{func.__name__}` requires `paddle` to be installed."
                        )
                elif backend_name == "flaxlib":
                    if not _is_flaxlib_available():
                        raise ImportError(
                            f"`{func.__name__}` requires `flaxlib` to be installed."
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


# mypy: disable-error-code="attr-defined"
class _LazyModule(ModuleType):  # pragma: no cover
    """Module class that surfaces all objects but only performs associated imports when the objects are requested.

    This class is inspired from the Hugging Face Transformers library:
    https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py#L1076
    """

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict,
        module_spec: Optional[Any] = None,
        extra_objects: Optional[dict] = None,
    ) -> None:
        """Initialize a _LazyModule instance."""
        super().__init__(name)

        self._modules = set(import_structure.keys())
        self._class_to_module = {}

        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key

        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(
            chain(*import_structure.values())
        )
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]

        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    def __dir__(self) -> Iterable[str]:
        """Allow for autocompletion in an IDE."""
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)

        return result

    def __getattr__(self, name: str) -> Any:
        """Retrieve the attribute `name` from the module.

        If the attribute is not directly found in the module's dictionary, it tries to lazily
        load the module or the class associated with that name. If the name refers to another module,
        it lazy-loads that module. If the name refers to a class or function,
        it lazy-loads the module containing that class or function and then retrieves
        the class or function from it.

        Raises:
            AttributeError: If the name cannot be resolved to any attribute, module, or class within this module.

        Args:
            name (str):
                The name of the attribute being accessed.

        Returns:
            Any: The attribute associated with the given name.
        """
        if name in self._objects:
            return self._objects[name]

        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)

        return value

    def _get_module(self, module_name: str) -> Any:
        """
        Import the module `module_name` and return it.

        Args:
            module_name (str):
                The name of the module to import.

        Returns:
            Any: The imported module.
        """
        try:
            return import_module("." + module_name, self.__name__)

        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the"
                f" following error (look up to see its traceback):\n{e}"
            ) from e

    def __reduce__(self) -> tuple:
        """Reduce the object for pickling."""
        return (self.__class__, (self._name, self.__file__, self._import_structure))
