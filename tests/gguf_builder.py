"""Minimal GGUF writer used to build synthetic fixtures in tests.

Kept independent of the production reader so that a bug in the reader cannot be
masked by the same bug in the writer.
"""

from __future__ import annotations

import struct
from typing import Any, Sequence

GGUF_MAGIC = b"GGUF"

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32 = 0, 1, 2, 3, 4, 5, 6
BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = 7, 8, 9, 10, 11, 12

_SCALAR_FMT = {
    UINT8: "<B",
    INT8: "<b",
    UINT16: "<H",
    INT16: "<h",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    BOOL: "<B",
    UINT64: "<Q",
    INT64: "<q",
    FLOAT64: "<d",
}


def encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_value(type_id: int, value: Any) -> bytes:
    if type_id == STRING:
        return encode_string(value)
    if type_id == ARRAY:
        element_type, items = value
        body = b"".join(encode_value(element_type, item) for item in items)
        return struct.pack("<IQ", element_type, len(items)) + body
    if type_id == BOOL:
        return struct.pack("<B", 1 if value else 0)
    return struct.pack(_SCALAR_FMT[type_id], value)


def build_gguf(
    kv: Sequence[tuple[str, int, Any]],
    *,
    version: int = 3,
    tensors: Sequence[tuple[str, Sequence[int], int]] = (),
    alignment: int = 32,
    magic: bytes = GGUF_MAGIC,
    tensor_count: int | None = None,
    kv_count: int | None = None,
    with_data: bool = True,
) -> bytes:
    """Serialise a complete GGUF file.

    ``tensor_count`` and ``kv_count`` override the declared counts so tests can
    build headers that lie about their own contents.
    """
    out = bytearray()
    out += magic
    out += struct.pack("<I", version)
    out += struct.pack("<Q", tensor_count if tensor_count is not None else len(tensors))
    out += struct.pack("<Q", kv_count if kv_count is not None else len(kv))
    for key, type_id, value in kv:
        out += encode_string(key)
        out += struct.pack("<I", type_id)
        out += encode_value(type_id, value)
    data_size = 0
    for name, dims, ggml_type in tensors:
        out += encode_string(name)
        out += struct.pack("<I", len(dims))
        for dim in dims:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", data_size)
        count = 1
        for dim in dims:
            count *= dim
        data_size += count * 4  # fixtures only use f32 tensors
    padding = (alignment - len(out) % alignment) % alignment
    out += b"\x00" * padding
    if with_data:
        out += b"\x00" * data_size
    return bytes(out)
