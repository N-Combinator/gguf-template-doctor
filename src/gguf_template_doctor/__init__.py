"""Offline GGUF chat-template inspector."""

from __future__ import annotations

__version__ = "0.1.0"

from .doctor import ChatTemplate, Finding, Report, Severity, diagnose, summarize
from .errors import (
    GgufError,
    GgufFileError,
    GgufMagicError,
    GgufMalformedError,
    GgufTruncatedError,
    GgufVersionError,
)
from .reader import GgufFile, GgufType, TensorInfo, parse_header

__all__ = [
    "ChatTemplate",
    "Finding",
    "GgufError",
    "GgufFile",
    "GgufFileError",
    "GgufMagicError",
    "GgufMalformedError",
    "GgufTruncatedError",
    "GgufType",
    "GgufVersionError",
    "Report",
    "Severity",
    "TensorInfo",
    "diagnose",
    "parse_header",
    "summarize",
    "__version__",
]
