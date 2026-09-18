"""Parser tests: real file, every value type, and every failure mode."""

from __future__ import annotations

import struct

import pytest
from gguf_builder import (
    ARRAY,
    BOOL,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    build_gguf,
)

from gguf_template_doctor import (
    GgufMagicError,
    GgufMalformedError,
    GgufTruncatedError,
    GgufVersionError,
    parse_header,
)
from gguf_template_doctor.errors import GgufError, GgufFileError


# --------------------------------------------------------------------------
# Criterion 1: a real GGUF file, not only homemade fixtures.
# --------------------------------------------------------------------------


def test_parses_real_gguf_header(real_gguf):
    gguf = parse_header(real_gguf)
    assert gguf.version == 3
    assert gguf.metadata["general.architecture"] == "qwen2"
    assert gguf.metadata["general.name"] == "qwen2.5-0.5b-instruct"
    # Values from the real file, covering string, uint32, float32 and bool.
    assert gguf.metadata["qwen2.block_count"] == 24
    assert gguf.metadata["qwen2.context_length"] == 32768
    assert gguf.metadata["qwen2.rope.freq_base"] == pytest.approx(1_000_000.0)
    assert gguf.metadata["tokenizer.ggml.add_bos_token"] is False
    assert gguf.metadata["tokenizer.ggml.model"] == "gpt2"


def test_real_gguf_arrays_and_template(real_gguf):
    gguf = parse_header(real_gguf)
    tokens = gguf.metadata["tokenizer.ggml.tokens"]
    token_types = gguf.metadata["tokenizer.ggml.token_type"]
    assert isinstance(tokens, list) and tokens[:3] == ["!", '"', "#"]
    assert isinstance(token_types, list) and token_types[:3] == [1, 1, 1]
    template = gguf.metadata["tokenizer.chat_template"]
    # The real Qwen2.5 template, unmodified.
    assert len(template) == 2509
    assert template.startswith("{%- if tools %}")
    assert "<|im_start|>" in template and "add_generation_prompt" in template


def test_real_gguf_tensor_table_and_alignment(real_gguf):
    gguf = parse_header(real_gguf)
    assert gguf.tensor_count == len(gguf.tensors) == 3
    assert gguf.tensors[0].name == "output.weight"
    assert gguf.data_offset % 32 == 0
    # Tensor-info records are the published ones: token_embd.weight keeps its real
    # [n_embd, n_vocab] shape, which is how the file still declares a 151936-token
    # vocabulary although only 64 tokens were kept.
    embedding = next(t for t in gguf.tensors if t.name == "token_embd.weight")
    assert embedding.dimensions == (896, 151936)
    # Header-only, so the payload those records point at is absent - one warning.
    assert len(gguf.warnings) == 1
    assert "tensor data" in gguf.warnings[0]


def test_real_gguf_records_its_provenance(real_gguf):
    """The fixture states where it came from, so it cannot be mistaken for synthetic."""
    gguf = parse_header(real_gguf)
    assert "huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF" in (
        gguf.metadata["general.source.url"]
    )
    assert gguf.metadata["testing.vocab_trimmed_from"] == 151936


def test_parses_second_real_gguf_header(nemo_gguf):
    """A second published file, different tokenizer and metadata layout."""
    gguf = parse_header(nemo_gguf)
    assert gguf.version == 3
    assert gguf.metadata["general.architecture"] == "llama"
    assert gguf.metadata["general.name"] == "Mistral Nemo Instruct 2407"
    assert gguf.metadata["tokenizer.ggml.pre"] == "tekken"
    assert gguf.metadata["general.languages"][:3] == ["en", "fr", "de"]
    assert gguf.metadata["llama.context_length"] == 1_024_000
    assert len(gguf.metadata["tokenizer.chat_template"]) == 3945
    assert "huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF" in (
        gguf.metadata["general.source.url"]
    )
    assert gguf.metadata["testing.vocab_trimmed_from"] == 131072


def test_second_real_gguf_reports_its_missing_tensor_data(nemo_gguf):
    """Header-only: metadata is complete, the payload is not, and that is a warning."""
    gguf = parse_header(nemo_gguf)
    assert gguf.tensor_count == len(gguf.tensors) == 3
    assert len(gguf.warnings) == 1
    assert "tensor data" in gguf.warnings[0]


# --------------------------------------------------------------------------
# Criterion 1: all KV value types from the specification.
# --------------------------------------------------------------------------


def test_all_scalar_types_round_trip(tmp_path):
    kv = [
        ("k.u8", UINT8, 255),
        ("k.i8", INT8, -128),
        ("k.u16", UINT16, 65535),
        ("k.i16", INT16, -32768),
        ("k.u32", UINT32, 4294967295),
        ("k.i32", INT32, -2147483648),
        ("k.f32", FLOAT32, 0.5),
        ("k.bool.true", BOOL, True),
        ("k.bool.false", BOOL, False),
        ("k.str", STRING, "hello"),
        ("k.u64", UINT64, 18446744073709551615),
        ("k.i64", INT64, -9223372036854775808),
        ("k.f64", FLOAT64, 1e-9),
    ]
    path = tmp_path / "scalars.gguf"
    path.write_bytes(build_gguf(kv))
    gguf = parse_header(path)
    assert gguf.metadata["k.u8"] == 255
    assert gguf.metadata["k.i8"] == -128
    assert gguf.metadata["k.u16"] == 65535
    assert gguf.metadata["k.i16"] == -32768
    assert gguf.metadata["k.u32"] == 4294967295
    assert gguf.metadata["k.i32"] == -2147483648
    assert gguf.metadata["k.f32"] == pytest.approx(0.5)
    assert gguf.metadata["k.bool.true"] is True
    assert gguf.metadata["k.bool.false"] is False
    assert gguf.metadata["k.str"] == "hello"
    assert gguf.metadata["k.u64"] == 18446744073709551615
    assert gguf.metadata["k.i64"] == -9223372036854775808
    assert gguf.metadata["k.f64"] == pytest.approx(1e-9)
    # Types are preserved alongside values.
    assert gguf.metadata_types["k.f64"] == FLOAT64


@pytest.mark.parametrize(
    "element_type,items",
    [
        (UINT8, [0, 1, 255]),
        (INT8, [-128, 0, 127]),
        (UINT16, [0, 65535]),
        (INT16, [-32768, 32767]),
        (UINT32, [0, 4294967295]),
        (INT32, [-2147483648, 2147483647]),
        (FLOAT32, [0.5, -1.5]),
        (UINT64, [0, 18446744073709551615]),
        (INT64, [-9223372036854775808, 0]),
        (FLOAT64, [1.5, -2.5]),
        (STRING, ["a", "bb", ""]),
    ],
)
def test_arrays_of_every_element_type(tmp_path, element_type, items):
    path = tmp_path / "array.gguf"
    path.write_bytes(build_gguf([("arr", ARRAY, (element_type, items))]))
    parsed = parse_header(path).metadata["arr"]
    assert len(parsed) == len(items)
    if element_type in (FLOAT32, FLOAT64):
        assert parsed == pytest.approx(items)
    else:
        assert parsed == items


def test_array_of_bool(tmp_path):
    path = tmp_path / "bools.gguf"
    path.write_bytes(
        build_gguf([("arr", ARRAY, (BOOL, [True, False, True, True]))])
    )
    assert parse_header(path).metadata["arr"] == [True, False, True, True]


def test_empty_array(tmp_path):
    path = tmp_path / "empty-array.gguf"
    path.write_bytes(build_gguf([("arr", ARRAY, (STRING, []))]))
    assert parse_header(path).metadata["arr"] == []


def test_nested_arrays(tmp_path):
    """The spec allows arrays of arrays; nested values must survive intact."""
    nested = (
        ARRAY,
        [
            (INT32, [1, 2, 3]),
            (INT32, []),
            (INT32, [-4]),
        ],
    )
    path = tmp_path / "nested.gguf"
    path.write_bytes(build_gguf([("nested", ARRAY, nested)]))
    assert parse_header(path).metadata["nested"] == [[1, 2, 3], [], [-4]]


def test_deeply_nested_arrays(tmp_path):
    inner = (STRING, ["deep"])
    level2 = (ARRAY, [inner])
    level3 = (ARRAY, [level2])
    path = tmp_path / "deep.gguf"
    path.write_bytes(build_gguf([("deep", ARRAY, level3)]))
    assert parse_header(path).metadata["deep"] == [[["deep"]]]


def test_array_nesting_beyond_limit_is_malformed(tmp_path):
    from gguf_template_doctor.reader import MAX_ARRAY_DEPTH

    value: tuple[int, list] = (STRING, ["x"])
    for _ in range(MAX_ARRAY_DEPTH + 2):
        value = (ARRAY, [value])
    path = tmp_path / "too-deep.gguf"
    path.write_bytes(build_gguf([("deep", ARRAY, value)]))
    with pytest.raises(GgufMalformedError, match="nesting"):
        parse_header(path)


def test_version_2_is_supported(tmp_path):
    path = tmp_path / "v2.gguf"
    path.write_bytes(build_gguf([("general.architecture", STRING, "llama")], version=2))
    assert parse_header(path).version == 2


def test_unicode_and_multiline_strings(tmp_path):
    text = "héllo\nвторая строка\n🙂 {{ x }}"
    path = tmp_path / "unicode.gguf"
    path.write_bytes(build_gguf([("k", STRING, text)]))
    assert parse_header(path).metadata["k"] == text


def test_non_utf8_string_does_not_crash(tmp_path):
    """Real writers have emitted invalid UTF-8; it must degrade, not explode."""
    body = bytearray()
    body += b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    key = b"k.bad"
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", STRING)
    bad = b"\xff\xfe invalid"
    body += struct.pack("<Q", len(bad)) + bad
    path = tmp_path / "bad-utf8.gguf"
    path.write_bytes(bytes(body))
    value = parse_header(path).metadata["k.bad"]
    assert "invalid" in value  # replacement characters, no exception


def test_duplicate_keys_warn_and_keep_last(tmp_path):
    path = tmp_path / "dup.gguf"
    path.write_bytes(
        build_gguf([("dup", STRING, "first"), ("dup", STRING, "second")])
    )
    gguf = parse_header(path)
    assert gguf.metadata["dup"] == "second"
    assert any("duplicate" in w for w in gguf.warnings)


def test_tensor_info_is_parsed(tmp_path):
    path = tmp_path / "tensors.gguf"
    path.write_bytes(
        build_gguf(
            [("general.architecture", STRING, "llama")],
            tensors=[("a.weight", (2, 3), 0), ("b.weight", (4,), 0)],
        )
    )
    gguf = parse_header(path)
    assert [t.name for t in gguf.tensors] == ["a.weight", "b.weight"]
    assert gguf.tensors[0].dimensions == (2, 3)
    assert gguf.tensors[1].dimensions == (4,)


def test_custom_alignment_is_honoured(tmp_path):
    path = tmp_path / "align.gguf"
    path.write_bytes(
        build_gguf(
            [("general.alignment", UINT32, 64)],
            tensors=[("a.weight", (2,), 0)],
            alignment=64,
        )
    )
    assert parse_header(path).data_offset % 64 == 0


def test_invalid_alignment_falls_back_with_warning(tmp_path):
    path = tmp_path / "bad-align.gguf"
    path.write_bytes(build_gguf([("general.alignment", UINT32, 0)]))
    gguf = parse_header(path)
    assert any("alignment" in w for w in gguf.warnings)
    assert gguf.data_offset % 32 == 0


def test_missing_tensor_data_is_a_warning_not_an_error(tmp_path):
    """A header-only file (the common way to share metadata) stays readable."""
    path = tmp_path / "header-only.gguf"
    path.write_bytes(
        build_gguf(
            [("general.architecture", STRING, "llama")],
            tensors=[("a.weight", (1024,), 0)],
            with_data=False,
        )
    )
    gguf = parse_header(path)
    assert gguf.metadata["general.architecture"] == "llama"
    assert any("tensor data is missing" in w for w in gguf.warnings)


# --------------------------------------------------------------------------
# Criterion 2: truncated and corrupt files.
# --------------------------------------------------------------------------


def test_truncated_real_file_raises_truncated_error(real_gguf, tmp_path):
    """MANDATORY: a cut-off real file is diagnosed, not crashed on."""
    full = real_gguf.read_bytes()
    path = tmp_path / "truncated.gguf"
    path.write_bytes(full[: len(full) // 2])
    with pytest.raises(GgufTruncatedError) as excinfo:
        parse_header(path)
    message = str(excinfo.value)
    assert "file ends inside" in message
    # The message must name what was being read when the file ran out.
    assert "metadata key" in message or "tensor" in message


@pytest.mark.parametrize("keep", [0, 1, 3, 4, 8, 16, 23, 24, 40, 100, 1000, 3000, 5000])
def test_truncation_at_every_stage_is_reported_cleanly(real_gguf, tmp_path, keep):
    """Cutting a real file at any offset yields a GgufError, never a crash."""
    full = real_gguf.read_bytes()
    path = tmp_path / f"cut-{keep}.gguf"
    path.write_bytes(full[:keep])
    with pytest.raises(GgufError):
        parse_header(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.gguf"
    path.write_bytes(b"")
    with pytest.raises(GgufTruncatedError, match="GGUF magic"):
        parse_header(path)


def test_bad_magic(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOTGGUF" + b"\x00" * 64)
    with pytest.raises(GgufMagicError, match="not a GGUF file"):
        parse_header(path)


def test_unsupported_version(tmp_path):
    path = tmp_path / "v1.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 1, 0, 0))
    with pytest.raises(GgufVersionError, match="unsupported GGUF version 1"):
        parse_header(path)


def test_unknown_value_type(tmp_path):
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    key = b"k"
    body += struct.pack("<Q", len(key)) + key + struct.pack("<I", 99)
    path = tmp_path / "bad-type.gguf"
    path.write_bytes(body)
    with pytest.raises(GgufMalformedError, match="unknown GGUF value type 99"):
        parse_header(path)


def test_unknown_array_element_type(tmp_path):
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    key = b"arr"
    body += struct.pack("<Q", len(key)) + key + struct.pack("<I", ARRAY)
    body += struct.pack("<IQ", 77, 2)
    path = tmp_path / "bad-elem.gguf"
    path.write_bytes(body)
    with pytest.raises(GgufMalformedError, match="unknown GGUF value type 77"):
        parse_header(path)


def test_absurd_string_length_is_rejected_without_allocating(tmp_path):
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    key = b"k"
    body += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
    body += struct.pack("<Q", 1 << 62)  # 4 exabytes
    path = tmp_path / "huge-string.gguf"
    path.write_bytes(body)
    with pytest.raises(GgufMalformedError, match="implausible length"):
        parse_header(path)


def test_absurd_array_length_is_rejected(tmp_path):
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    key = b"arr"
    body += struct.pack("<Q", len(key)) + key + struct.pack("<I", ARRAY)
    body += struct.pack("<IQ", INT32, 1 << 40)
    path = tmp_path / "huge-array.gguf"
    path.write_bytes(body)
    with pytest.raises(GgufMalformedError, match="elements"):
        parse_header(path)


def test_absurd_kv_count_is_rejected(tmp_path):
    path = tmp_path / "many-kv.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 40))
    with pytest.raises(GgufMalformedError, match="metadata entries"):
        parse_header(path)


def test_absurd_tensor_count_is_rejected(tmp_path):
    path = tmp_path / "many-tensors.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1 << 40, 0))
    with pytest.raises(GgufMalformedError, match="tensors"):
        parse_header(path)


def test_kv_count_higher_than_contents_is_truncation(tmp_path):
    path = tmp_path / "lying-kv.gguf"
    path.write_bytes(build_gguf([("k", STRING, "v")], kv_count=5))
    with pytest.raises(GgufTruncatedError):
        parse_header(path)


def test_too_many_tensor_dimensions(tmp_path):
    body = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    name = b"t"
    body += struct.pack("<Q", len(name)) + name + struct.pack("<I", 99)
    path = tmp_path / "bad-dims.gguf"
    path.write_bytes(body)
    with pytest.raises(GgufMalformedError, match="dimensions"):
        parse_header(path)


def test_missing_file(tmp_path):
    with pytest.raises(GgufFileError, match="cannot stat"):
        parse_header(tmp_path / "nope.gguf")


def test_directory_instead_of_file(tmp_path):
    with pytest.raises(GgufFileError):
        parse_header(tmp_path)


def test_all_reader_errors_are_gguf_errors(tmp_path):
    """Nothing escapes as a bare struct/OSError, which would become a traceback."""
    path = tmp_path / "x.gguf"
    for payload in (
        b"",
        b"GG",
        b"GGUF",
        b"GGUF" + struct.pack("<I", 3),
        b"GGUF" + struct.pack("<IQ", 3, 1),
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + b"\x01",
        b"GGUF" + struct.pack("<IQQ", 3, 3, 0),
        b"\x00" * 24,
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2) + b"ab",
    ):
        path.write_bytes(payload)
        with pytest.raises(GgufError):
            parse_header(path)
