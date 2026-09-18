"""Error types for gguf-template-doctor.

Every failure the tool can anticipate is raised as a :class:`GgufError` and turned
into a one-line message plus exit code 2 by the CLI.  Nothing should ever reach the
user as a traceback.
"""

from __future__ import annotations


class GgufError(Exception):
    """Base class for all expected, reportable failures."""


class GgufFileError(GgufError):
    """The file could not be opened or read at all."""


class GgufMagicError(GgufError):
    """The file does not start with the GGUF magic bytes."""


class GgufVersionError(GgufError):
    """The GGUF version is one this parser does not implement."""


class GgufTruncatedError(GgufError):
    """The file ended in the middle of a structure the header promised."""


class GgufMalformedError(GgufError):
    """The header is self-inconsistent: bad type tag, absurd length, bad UTF-8."""
