"""Pure-stdlib reader for the GGUF header and metadata key/value store.

Only the header is read: magic, version, tensor-info table and the metadata KV
store.  Tensor payloads are never touched, so inspecting a 40 GB model costs the
same as inspecting a 5 KB one.

Reference: the GGUF specification (ggml/docs/gguf.md), value types 0-12.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

from .errors import (
    GgufError,
    GgufFileError,
    GgufMagicError,
    GgufMalformedError,
    GgufTruncatedError,
    GgufVersionError,
)

GGUF_MAGIC = b"GGUF"

#: Versions whose header layout this parser implements.  v1 used 32-bit counts
#: and is not produced by any current tool, so it is rejected explicitly rather
#: than silently mis-parsed.
SUPPORTED_VERSIONS = (2, 3)

#: Guard against a corrupt length field making us allocate the whole machine.
#: No legitimate GGUF metadata string or array comes near these.
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_LEN = 1 << 28
MAX_KV_COUNT = 1 << 22
MAX_TENSOR_COUNT = 1 << 22
MAX_TENSOR_DIMS = 8


class GgufType(IntEnum):
    """GGUF metadata value types, as numbered by the specification."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


#: struct format and width for every scalar type.
_SCALARS: dict[int, tuple[str, int]] = {
    GgufType.UINT8: ("<B", 1),
    GgufType.INT8: ("<b", 1),
    GgufType.UINT16: ("<H", 2),
    GgufType.INT16: ("<h", 2),
    GgufType.UINT32: ("<I", 4),
    GgufType.INT32: ("<i", 4),
    GgufType.FLOAT32: ("<f", 4),
    GgufType.BOOL: ("<B", 1),
    GgufType.UINT64: ("<Q", 8),
    GgufType.INT64: ("<q", 8),
    GgufType.FLOAT64: ("<d", 8),
}

#: How deep nested arrays may go before we call the file malformed.  The spec
#: allows arrays of arrays; real files use at most one level, but a corrupt type
#: tag can otherwise drive unbounded recursion.
MAX_ARRAY_DEPTH = 8


@dataclass
class TensorInfo:
    """One entry of the tensor-info table (name, shape, quantisation, offset)."""

    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass
class GgufFile:
    """Parsed GGUF header."""

    path: Path
    version: int
    tensor_count: int
    kv_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_types: dict[str, int] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    #: Offset where the aligned tensor data section begins.
    data_offset: int = 0
    #: Non-fatal observations made while parsing (duplicate keys, short data
    #: section, ...).  A header that yields warnings is still usable.
    warnings: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)


class _Cursor:
    """Sequential reader over a file object that raises precise GGUF errors."""

    def __init__(self, stream: BinaryIO, path: Path, size: int | None = None) -> None:
        self._stream = stream
        self._path = path
        self._size = size
        self.offset = 0

    def read(self, n: int, what: str) -> bytes:
        if n < 0:
            raise GgufMalformedError(f"negative length {n} while reading {what}")
        # Refuse to allocate a buffer larger than the file could possibly fill.
        # Only a wildly oversized claim is rejected up front; an ordinary short
        # read falls through so the message names the structure that was cut off.
        if self._size is not None and self.offset + n > self._size:
            raise GgufTruncatedError(
                f"file ends inside {what}: wanted {n} bytes at offset "
                f"{self.offset} but the file is only {self._size} bytes"
            )
        data = self._stream.read(n)
        if len(data) != n:
            raise GgufTruncatedError(
                f"file ends inside {what}: wanted {n} bytes at offset "
                f"{self.offset}, got {len(data)}"
            )
        self.offset += n
        return data

    def scalar(self, fmt: str, width: int, what: str) -> Any:
        return struct.unpack(fmt, self.read(width, what))[0]

    def u32(self, what: str) -> int:
        return self.scalar("<I", 4, what)

    def u64(self, what: str) -> int:
        return self.scalar("<Q", 8, what)

    def string(self, what: str) -> str:
        length = self.u64(f"{what} length")
        if length > MAX_STRING_BYTES:
            raise GgufMalformedError(
                f"{what} claims an implausible length of {length} bytes at offset "
                f"{self.offset}"
            )
        raw = self.read(length, what)
        # GGUF strings are specified as UTF-8 but real-world writers have emitted
        # lone surrogates and latin-1 bytes.  Surface that as a value we can still
        # inspect rather than aborting the whole file.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")


def _read_value(cur: _Cursor, type_id: int, what: str, depth: int = 0) -> Any:
    if type_id == GgufType.STRING:
        return cur.string(what)
    if type_id == GgufType.ARRAY:
        if depth >= MAX_ARRAY_DEPTH:
            raise GgufMalformedError(
                f"{what}: array nesting deeper than {MAX_ARRAY_DEPTH} levels"
            )
        elem_type = cur.u32(f"{what} element type")
        count = cur.u64(f"{what} element count")
        if count > MAX_ARRAY_LEN:
            raise GgufMalformedError(
                f"{what} claims {count} elements at offset {cur.offset}"
            )
        _check_type(elem_type, f"{what} element type")
        if elem_type == GgufType.BOOL:
            return [b != 0 for b in cur.read(count, f"{what} contents")]
        if elem_type in _SCALARS:
            fmt, width = _SCALARS[elem_type]
            raw = cur.read(count * width, f"{what} contents")
            # One bulk unpack instead of `count` reads: a 151936-token vocab
            # array is the common case and this keeps it cheap.
            return list(struct.unpack(f"<{count}{fmt[1]}", raw))
        return [
            _read_value(cur, elem_type, f"{what}[{i}]", depth + 1) for i in range(count)
        ]
    if type_id == GgufType.BOOL:
        return cur.scalar("<B", 1, what) != 0
    fmt, width = _SCALARS[type_id]
    return cur.scalar(fmt, width, what)


def _check_type(type_id: int, what: str) -> None:
    if type_id not in set(GgufType):
        raise GgufMalformedError(f"{what}: unknown GGUF value type {type_id}")


#: Per ggml type: (block size in elements, bytes per block).  Quantised types
#: store whole blocks, so a tensor's byte size is
#: elements / block_size * type_size.  Values follow the ggml type table in the
#: GGUF specification.  A type absent from this table has an unknown element
#: size: see :func:`_tensor_data_size`, which refuses to guess one.
_GGML_TYPE_LAYOUT: dict[int, tuple[int, int]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 40),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
}


def _tensor_data_size(tensors: list[TensorInfo]) -> tuple[int | None, list[int]]:
    """Size of the tensor data section, and the ggml types that blocked it.

    Returns ``(total, unknown_types)``.  ``total`` is ``None`` when any tensor
    uses a ggml type this tool does not know: an unknown type has an unknown
    element size, and guessing one would turn a missing-payload check into a
    fabricated number.  In that case the caller reports that the size could not
    be computed instead of reporting a size.
    """
    total = 0
    unknown: list[int] = []
    for tensor in tensors:
        layout = _GGML_TYPE_LAYOUT.get(tensor.ggml_type)
        if layout is None:
            if tensor.ggml_type not in unknown:
                unknown.append(tensor.ggml_type)
            continue
        elements = 1
        for dim in tensor.dimensions:
            elements *= dim
        block_size, type_size = layout
        blocks = -(-elements // block_size)  # ceil: a partial block is stored whole
        total += blocks * type_size
    if unknown:
        return None, unknown
    return total, unknown


def parse_header(path: str | Path) -> GgufFile:
    """Parse the GGUF header at *path*.

    Raises a :class:`~gguf_template_doctor.errors.GgufError` subclass for any
    unreadable, truncated or malformed file.
    """
    path = Path(path)
    try:
        size: int | None = path.stat().st_size
    except OSError as exc:
        raise GgufFileError(f"cannot stat {path}: {exc.strerror or exc}") from exc
    try:
        with path.open("rb") as stream:
            return _parse_stream(stream, path, size)
    except GgufError:
        # Already a precise, reportable diagnosis - do not rewrap it.
        raise
    except OSError as exc:
        raise GgufFileError(f"cannot read {path}: {exc.strerror or exc}") from exc


def _parse_stream(stream: BinaryIO, path: Path, size: int | None) -> GgufFile:
    cur = _Cursor(stream, path, size)

    magic = cur.read(4, "GGUF magic")
    if magic != GGUF_MAGIC:
        raise GgufMagicError(
            f"{path} is not a GGUF file: expected magic {GGUF_MAGIC!r}, got {magic!r}"
        )
    version = cur.u32("GGUF version")
    if version not in SUPPORTED_VERSIONS:
        raise GgufVersionError(
            f"{path}: unsupported GGUF version {version} "
            f"(this tool reads versions {', '.join(map(str, SUPPORTED_VERSIONS))})"
        )
    tensor_count = cur.u64("tensor count")
    kv_count = cur.u64("metadata key/value count")
    if tensor_count > MAX_TENSOR_COUNT:
        raise GgufMalformedError(
            f"{path}: header claims {tensor_count} tensors, which is not plausible"
        )
    if kv_count > MAX_KV_COUNT:
        raise GgufMalformedError(
            f"{path}: header claims {kv_count} metadata entries, "
            "which is not plausible"
        )

    result = GgufFile(
        path=path,
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
    )

    for index in range(kv_count):
        key = cur.string(f"metadata key #{index}")
        type_id = cur.u32(f"type of metadata key {key!r}")
        _check_type(type_id, f"type of metadata key {key!r}")
        value = _read_value(cur, type_id, f"value of metadata key {key!r}")
        if key in result.metadata:
            result.warnings.append(
                f"duplicate metadata key {key!r}; keeping the last occurrence"
            )
        result.metadata[key] = value
        result.metadata_types[key] = type_id

    for index in range(tensor_count):
        name = cur.string(f"tensor name #{index}")
        n_dims = cur.u32(f"dimension count of tensor {name!r}")
        if n_dims > MAX_TENSOR_DIMS:
            raise GgufMalformedError(
                f"tensor {name!r} claims {n_dims} dimensions (maximum is "
                f"{MAX_TENSOR_DIMS})"
            )
        dims = tuple(cur.u64(f"dimension of tensor {name!r}") for _ in range(n_dims))
        ggml_type = cur.u32(f"type of tensor {name!r}")
        offset = cur.u64(f"offset of tensor {name!r}")
        result.tensors.append(TensorInfo(name, dims, ggml_type, offset))

    alignment = result.metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        result.warnings.append(
            f"general.alignment is {alignment!r}; using the default of 32"
        )
        alignment = 32
    padding = (alignment - cur.offset % alignment) % alignment
    result.data_offset = cur.offset + padding

    # The header is complete at this point, so a short or absent data section is
    # a warning rather than an error: the metadata and template are fully
    # readable, which is all this tool needs.  Header-only files are a normal way
    # to share model metadata.
    if size is not None:
        data_size, unknown_types = _tensor_data_size(result.tensors)
        if data_size is None:
            listed = ", ".join(str(t) for t in sorted(unknown_types))
            result.warnings.append(
                f"tensor data section size not computed: unknown ggml type(s) "
                f"{listed}; cannot tell whether the tensor data is present"
            )
        else:
            expected_end = result.data_offset + data_size
            if size < expected_end:
                result.warnings.append(
                    f"file ends at {size} bytes but the tensor data section runs to "
                    f"{expected_end}; header is complete but tensor data is missing "
                    "or truncated"
                )

    return result
