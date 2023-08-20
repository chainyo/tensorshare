"""Logging module for tensorshare."""

import logging


class TensorShareLogger(logging.Logger):
    """TensorShare logger class."""

    _warning_cache = set()

    def __init__(self, name: str, *args, **kwargs) -> None:
        """Initialize the logger."""
        super().__init__(name, *args, **kwargs)

    def warning_once(self, *args, **kwargs) -> None:
        """A warning method that only logs once."""
        hash_key = hash(str(args) + str(kwargs))
        if hash_key not in TensorShareLogger._warning_cache:
            self.warning(*args, **kwargs)
            TensorShareLogger._warning_cache.add(hash_key)


logging.setLoggerClass(TensorShareLogger)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
