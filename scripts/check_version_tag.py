#!/usr/bin/env python3
"""Check that a release tag agrees with the version the package will ship.

Run by ``.github/workflows/release.yml`` before anything is built, so a tag that
disagrees with ``pyproject.toml`` fails the release instead of publishing a wheel
whose metadata says something else.

Two version sources are checked, because the package has two: ``pyproject.toml``
decides what goes into the wheel metadata, and ``gguf_template_doctor.__version__``
is what ``gguf-template-doctor --version`` prints.  A release in which those
disagree ships a binary that lies about itself.

Usage::

    python scripts/check_version_tag.py v1.2.3

Exit codes: 0 everything agrees, 1 a mismatch or a malformed tag, 2 a version
could not be read at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Release tags are exactly ``vX.Y.Z``.  Anything else is a mistake worth stopping
#: for rather than guessing at.
TAG_RE = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")

_PYPROJECT_VERSION_RE = re.compile(
    r"^version\s*=\s*[\"'](?P<version>[^\"']+)[\"']", re.MULTILINE
)
_DUNDER_VERSION_RE = re.compile(
    r"^__version__\s*=\s*[\"'](?P<version>[^\"']+)[\"']", re.MULTILINE
)


class VersionReadError(Exception):
    """A version could not be read from a file it was expected to be in."""


def _read_text(path: Path) -> str:
    """Read a source file without turning an encoding quirk into a traceback."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise VersionReadError(f"{path}: file not found") from exc
    except UnicodeDecodeError as exc:
        raise VersionReadError(f"{path}: not valid UTF-8 ({exc.reason})") from exc
    except OSError as exc:
        raise VersionReadError(f"{path}: {exc.strerror or exc}") from exc


def read_pyproject_version(root: Path = ROOT) -> str:
    """Return the ``version`` declared in ``[project]`` of ``pyproject.toml``."""
    path = root / "pyproject.toml"
    text = _read_text(path)
    # The file has exactly one top-level ``version =``; the build-system table
    # above it declares requirements, not a version.
    match = _PYPROJECT_VERSION_RE.search(text)
    if match is None:
        raise VersionReadError(f"{path}: no top-level version = \"...\" found")
    return match.group("version")


def read_package_version(root: Path = ROOT) -> str:
    """Return ``__version__`` from the package's ``__init__.py``."""
    path = root / "src" / "gguf_template_doctor" / "__init__.py"
    text = _read_text(path)
    match = _DUNDER_VERSION_RE.search(text)
    if match is None:
        raise VersionReadError(f"{path}: no __version__ = \"...\" found")
    return match.group("version")


def version_from_tag(tag: str) -> str:
    """Return the bare version in a ``vX.Y.Z`` tag, or raise ``ValueError``."""
    match = TAG_RE.match(tag)
    if match is None:
        raise ValueError(f"tag {tag!r} is not of the form vX.Y.Z")
    return match.group("version")


def check(tag: str, root: Path = ROOT) -> list[str]:
    """Return the list of disagreements between *tag* and the declared versions.

    An empty list means the release may proceed.
    """
    problems: list[str] = []
    try:
        wanted = version_from_tag(tag)
    except ValueError as exc:
        return [str(exc)]

    for label, reader in (
        ("pyproject.toml", read_pyproject_version),
        ("gguf_template_doctor.__version__", read_package_version),
    ):
        found = reader(root)
        if found != wanted:
            problems.append(
                f"{label} is {found!r} but the tag says {wanted!r}"
            )
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} vX.Y.Z", file=sys.stderr)
        return 1
    tag = argv[1]
    try:
        problems = check(tag)
    except VersionReadError as exc:
        print(f"cannot read the declared version: {exc}", file=sys.stderr)
        return 2

    if problems:
        print(f"release tag {tag} does not match the packaged version:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "Bump the version in pyproject.toml and src/gguf_template_doctor/"
            "__init__.py, or retag, then try again.",
            file=sys.stderr,
        )
        return 1

    print(f"release tag {tag} matches the packaged version")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(main(sys.argv))
