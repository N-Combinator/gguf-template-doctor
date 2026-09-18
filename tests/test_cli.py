"""CLI tests: exit codes, readable errors, no tracebacks (criteria 2 and 4)."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest
from gguf_builder import ARRAY, FLOAT32, FLOAT64, STRING, UINT32, build_gguf

from gguf_template_doctor.cli import (
    EXIT_FINDINGS,
    EXIT_OK,
    EXIT_UNREADABLE,
    main,
)

GOOD_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
VOCAB = ["<|im_start|>", "<|im_end|>", "a", "b"]


def run(*argv):
    """Invoke the CLI in-process, returning (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def good_model(tmp_path):
    path = tmp_path / "good.gguf"
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "llama"),
                ("general.name", STRING, "good-model"),
                ("tokenizer.ggml.model", STRING, "gpt2"),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, VOCAB)),
                ("tokenizer.ggml.eos_token_id", UINT32, 1),
                ("tokenizer.chat_template", STRING, GOOD_TEMPLATE),
            ]
        )
    )
    return path


@pytest.fixture
def broken_model(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "llama"),
                ("tokenizer.ggml.model", STRING, "gpt2"),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, VOCAB)),
                ("tokenizer.ggml.eos_token_id", UINT32, 1),
                ("tokenizer.chat_template", STRING, "{% for m in messages %}{{ m }}"),
            ]
        )
    )
    return path


# --------------------------------------------------------------------------
# Exit code 0: healthy file.
# --------------------------------------------------------------------------


def test_healthy_model_exits_zero(good_model):
    code, out, err = run(str(good_model))
    assert code == EXIT_OK
    assert "findings: none" in out
    assert err == ""


def test_real_file_exits_zero(real_gguf):
    code, out, err = run(str(real_gguf))
    assert code == EXIT_OK, out + err
    assert "qwen2" in out
    assert "chat template(s): default" in out


def test_published_nemo_header_reports_no_errors(nemo_gguf):
    """Regression: a literal `}}` in the template used to be an ERROR finding."""
    code, out, err = run(str(nemo_gguf))
    assert "unbalanced" not in out
    assert "0 error(s)" in out
    assert code == EXIT_FINDINGS  # the remaining no-generation-prompt warning
    assert err == ""


def test_show_template_prints_the_real_template(real_gguf, qwen_template):
    code, out, _ = run(str(real_gguf), "--show-template")
    assert code == EXIT_OK
    assert qwen_template in out


def test_list_metadata_lists_real_keys(real_gguf):
    code, out, _ = run(str(real_gguf), "--list-metadata")
    assert code == EXIT_OK
    assert "general.architecture (string)" in out
    assert "tokenizer.ggml.tokens (array)" in out
    assert "qwen2.block_count (uint32)" in out


# --------------------------------------------------------------------------
# Exit code 1: template problems found.
# --------------------------------------------------------------------------


def test_broken_template_exits_one(broken_model):
    code, out, _ = run(str(broken_model))
    assert code == EXIT_FINDINGS
    assert "unbalanced-block" in out
    assert "error:" in out


def test_missing_template_exits_one(tmp_path):
    path = tmp_path / "no-template.gguf"
    path.write_bytes(build_gguf([("general.architecture", STRING, "llama")]))
    code, out, _ = run(str(path))
    assert code == EXIT_FINDINGS
    assert "no-chat-template" in out


def test_eos_id_past_the_vocabulary_exits_one(tmp_path):
    """A file whose eos id points past its own vocabulary must not look healthy."""
    path = tmp_path / "bad-eos.gguf"
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "llama"),
                ("tokenizer.ggml.model", STRING, "gpt2"),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, ["a", "b", "c"])),
                ("tokenizer.ggml.eos_token_id", UINT32, 999),
                ("tokenizer.chat_template", STRING, GOOD_TEMPLATE),
            ]
        )
    )
    code, out, _ = run(str(path))
    assert code == EXIT_FINDINGS
    assert "token-id-out-of-range" in out
    assert "eos_token_id is 999" in out


def test_strict_turns_info_into_failure(real_gguf):
    assert run(str(real_gguf))[0] == EXIT_OK
    code, out, _ = run(str(real_gguf), "--strict")
    assert code == EXIT_FINDINGS
    assert "partial-vocabulary" in out


# --------------------------------------------------------------------------
# Criterion 2: exit code 2 with a readable error, never a traceback.
# --------------------------------------------------------------------------


def test_truncated_file_exits_two_with_clear_error(real_gguf, tmp_path):
    """MANDATORY test: truncated input -> exit 2, readable message, no traceback."""
    full = real_gguf.read_bytes()
    path = tmp_path / "truncated.gguf"
    path.write_bytes(full[:3000])
    code, out, err = run(str(path))
    assert code == EXIT_UNREADABLE
    assert out == ""
    assert err.startswith("gguf-template-doctor: error: ")
    assert "file ends inside" in err
    assert "Traceback" not in err
    assert err.count("\n") == 1  # exactly one line


@pytest.mark.parametrize("keep", [0, 4, 12, 24, 64, 512, 2048, 4096])
def test_truncation_at_any_offset_exits_two(real_gguf, tmp_path, keep):
    path = tmp_path / f"cut-{keep}.gguf"
    path.write_bytes(real_gguf.read_bytes()[:keep])
    code, out, err = run(str(path))
    assert code == EXIT_UNREADABLE
    assert "Traceback" not in err
    assert err.strip()


def test_corrupt_bytes_in_the_middle_exit_two(real_gguf, tmp_path):
    """Flipping a length field to garbage must be reported, not crash."""
    data = bytearray(real_gguf.read_bytes())
    data[24:32] = (1 << 62).to_bytes(8, "little")  # first key's length
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(bytes(data))
    code, _, err = run(str(path))
    assert code == EXIT_UNREADABLE
    assert "Traceback" not in err


def test_not_a_gguf_file_exits_two(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("this is not a model\n")
    code, _, err = run(str(path))
    assert code == EXIT_UNREADABLE
    assert "not a GGUF file" in err


def test_empty_file_exits_two(tmp_path):
    path = tmp_path / "empty.gguf"
    path.write_bytes(b"")
    code, _, err = run(str(path))
    assert code == EXIT_UNREADABLE
    assert "Traceback" not in err


def test_missing_file_exits_two(tmp_path):
    code, _, err = run(str(tmp_path / "absent.gguf"))
    assert code == EXIT_UNREADABLE
    assert "No such file" in err


def test_directory_exits_two(tmp_path):
    code, _, err = run(str(tmp_path))
    assert code == EXIT_UNREADABLE
    assert "Traceback" not in err


def test_unreadable_file_exits_two(tmp_path):
    path = tmp_path / "locked.gguf"
    path.write_bytes(b"GGUF")
    path.chmod(0o000)
    try:
        code, _, err = run(str(path))
    finally:
        path.chmod(0o644)
    # Running as root defeats the permission bits; then it is merely truncated.
    assert code == EXIT_UNREADABLE
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# JSON output.
# --------------------------------------------------------------------------


def test_json_output_is_valid_and_complete(real_gguf):
    code, out, _ = run(str(real_gguf), "--json")
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["architecture"] == "qwen2"
    assert payload["gguf_version"] == 3
    assert payload["ok"] is True
    assert payload["templates"][0]["name"] == "default"
    assert payload["templates"][0]["length"] == 2509
    assert {f["code"] for f in payload["findings"]} == {"partial-vocabulary"}


def test_json_output_for_broken_template(broken_model):
    code, out, _ = run(str(broken_model), "--json")
    assert code == EXIT_FINDINGS
    payload = json.loads(out)
    assert payload["ok"] is False
    assert any(
        f["code"] == "unbalanced-block" and f["severity"] == "error"
        for f in payload["findings"]
    )


def test_json_show_template_includes_source(real_gguf, qwen_template):
    code, out, _ = run(str(real_gguf), "--json", "--show-template")
    payload = json.loads(out)
    assert payload["template_sources"]["default"] == qwen_template


def test_json_list_metadata_omits_huge_arrays(real_gguf):
    code, out, _ = run(str(real_gguf), "--json", "--list-metadata")
    payload = json.loads(out)
    assert payload["metadata"]["general.architecture"] == "qwen2"


# --------------------------------------------------------------------------
# Scope: --compare-known is intentionally absent (criterion 3).
# --------------------------------------------------------------------------


def test_compare_known_flag_does_not_exist(real_gguf):
    """No reference sources were specified for the issue, so the flag is not built."""
    with pytest.raises(SystemExit) as excinfo:
        main([str(real_gguf), "--compare-known"], out=io.StringIO(), err=io.StringIO())
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# The installed entry point behaves like the in-process one.
# --------------------------------------------------------------------------


def test_module_invocation_exits_two_on_truncated_file(real_gguf, tmp_path):
    """End-to-end through a real interpreter: no traceback reaches the user."""
    path = tmp_path / "cut.gguf"
    path.write_bytes(real_gguf.read_bytes()[:1500])
    result = subprocess.run(
        [sys.executable, "-m", "gguf_template_doctor.cli", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert result.stderr.startswith("gguf-template-doctor: error: ")


def test_module_invocation_succeeds_on_real_file(real_gguf):
    result = subprocess.run(
        [sys.executable, "-m", "gguf_template_doctor.cli", str(real_gguf)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "qwen2" in result.stdout


@pytest.fixture
def non_finite_model(tmp_path):
    """A model whose float metadata holds NaN and +/-Infinity.

    Legal GGUF: FLOAT32/FLOAT64 are IEEE-754, so these are representable and
    real files can carry them.
    """
    path = tmp_path / "nonfinite.gguf"
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "llama"),
                ("general.name", STRING, "non-finite"),
                ("tokenizer.ggml.model", STRING, "gpt2"),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, VOCAB)),
                ("tokenizer.ggml.eos_token_id", UINT32, 1),
                ("tokenizer.chat_template", STRING, GOOD_TEMPLATE),
                ("llama.rope.freq_base", FLOAT32, float("nan")),
                ("llama.attention.clamp_kqv", FLOAT64, float("inf")),
                ("llama.expert_weights_scale", FLOAT32, float("-inf")),
                ("llama.f32.scales", ARRAY, (FLOAT32, [1.5, float("nan")])),
            ]
        )
    )
    return path


def test_json_output_with_non_finite_floats_is_strict_json(non_finite_model):
    """Bare NaN/Infinity literals are not JSON; a strict parser must accept us."""
    code, out, _ = run(str(non_finite_model), "--json", "--list-metadata")
    assert code == EXIT_OK

    # json.loads accepts the bare literals by default, so reject them explicitly.
    def _no_constants(name):
        raise AssertionError(f"non-JSON literal {name!r} in output")

    payload = json.loads(out, parse_constant=_no_constants)
    metadata = payload["metadata"]
    assert metadata["llama.rope.freq_base"] == "NaN"
    assert metadata["llama.attention.clamp_kqv"] == "Infinity"
    assert metadata["llama.expert_weights_scale"] == "-Infinity"
    # Finite values inside arrays stay numbers; only the non-finite one converts.
    assert metadata["llama.f32.scales"] == [1.5, "NaN"]


def test_json_output_keeps_finite_floats_as_numbers(non_finite_model):
    code, out, _ = run(str(non_finite_model), "--json", "--list-metadata")
    assert code == EXIT_OK
    payload = json.loads(out)
    assert isinstance(payload["metadata"]["llama.f32.scales"][0], float)


# --------------------------------------------------------------------------
# --list-metadata --json reports a long array's length; it never drops the key.
# --------------------------------------------------------------------------


def test_json_metadata_reports_long_list_length_instead_of_dropping_it(tmp_path):
    tokens = [f"t{i}" for i in range(500)]
    path = tmp_path / "long.gguf"
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "llama"),
                ("tokenizer.chat_template", STRING, GOOD_TEMPLATE),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, tokens)),
                ("tokenizer.ggml.eos_token_id", UINT32, 1),
            ]
        )
    )
    code, out, _ = run(str(path), "--json", "--list-metadata")
    assert code in (EXIT_OK, EXIT_FINDINGS)
    metadata = json.loads(out)["metadata"]
    assert "tokenizer.ggml.tokens" in metadata, "the key must not be dropped"
    entry = metadata["tokenizer.ggml.tokens"]
    assert entry["length"] == 500
    assert entry["truncated"] is True
    assert entry["items"] == tokens[:64]


def test_json_metadata_keeps_short_lists_verbatim(tmp_path, good_model):
    code, out, _ = run(str(good_model), "--json", "--list-metadata")
    assert code in (EXIT_OK, EXIT_FINDINGS)
    metadata = json.loads(out)["metadata"]
    assert metadata["tokenizer.ggml.tokens"] == VOCAB


# --------------------------------------------------------------------------
# A narrow stdout encoding must degrade the report, not raise.
# --------------------------------------------------------------------------


class _AsciiOut(io.StringIO):
    """Stands in for a stdout opened in a narrow encoding (e.g. LC_ALL=C)."""

    encoding = "ascii"

    def write(self, text):
        text.encode(self.encoding)  # raises UnicodeEncodeError like a real stream
        return super().write(text)


def _non_ascii_model(tmp_path, name="nonascii.gguf"):
    template = GOOD_TEMPLATE + "{{ 'Ответ — ' }}"
    path = tmp_path / name
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "llama"),
                ("general.name", STRING, "модель"),
                ("tokenizer.chat_template", STRING, template),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, VOCAB)),
                ("tokenizer.ggml.eos_token_id", UINT32, 1),
            ]
        )
    )
    return path


def test_text_report_survives_a_non_utf8_stdout(tmp_path):
    path = _non_ascii_model(tmp_path)
    out, err = _AsciiOut(), io.StringIO()
    code = main([str(path), "--show-template"], out=out, err=err)
    assert code in (EXIT_OK, EXIT_FINDINGS)
    text = out.getvalue()
    assert "findings" in text
    assert "?" in text  # the unencodable characters were replaced, not fatal


def test_json_report_survives_a_non_utf8_stdout(tmp_path):
    path = _non_ascii_model(tmp_path)
    out, err = _AsciiOut(), io.StringIO()
    code = main([str(path), "--json", "--list-metadata"], out=out, err=err)
    assert code in (EXIT_OK, EXIT_FINDINGS)
    assert json.loads(out.getvalue())["metadata"]["general.architecture"] == "llama"


def test_unreadable_file_error_survives_a_non_utf8_stderr(tmp_path):
    path = tmp_path / "модель.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 32)
    out, err = io.StringIO(), _AsciiOut()
    code = main([str(path)], out=out, err=err)
    assert code == EXIT_UNREADABLE
    assert "error:" in err.getvalue()
