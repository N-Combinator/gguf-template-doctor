"""CLI tests: exit codes, readable errors, no tracebacks (criteria 2 and 4)."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest
from gguf_builder import ARRAY, STRING, UINT32, build_gguf

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
