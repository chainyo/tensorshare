# mypy: disable-error-code="attr-defined"
"""`tensorshare `: 🤝 Trade any tensors over the network"""

import sys

from .import_utils import _LazyModule

__author__ = "Thomas Chaigneau <t.chaigneau.tc@gmail.com>"
__version__ = "0.0.5"

_import_structure = {
    "import_utils": ["require_backend"],
    "client": ["TensorShareClient"],
    "schema": ["DefaultResponse", "TensorShare", "TensorShareServer"],
    "serialization": ["Backend", "TensorProcessor", "TensorType"],
    "server": ["create_tensorshare_router", "create_async_tensorshare_router"],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,  # type: ignore
    extra_objects={"__version__": __version__},
)
