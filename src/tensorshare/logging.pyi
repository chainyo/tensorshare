"""Stub file for tensorshare.logging."""

from logging import Logger
from typing import Any

class TensorShareLogger(Logger):
    def warning_once(self, *args: Any, **kwargs: Any) -> None: ...

logger: TensorShareLogger
