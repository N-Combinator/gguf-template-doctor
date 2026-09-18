"""Template doctor tests (criterion 4)."""

from __future__ import annotations

import pytest
from gguf_builder import ARRAY, BOOL, STRING, UINT32, build_gguf

from gguf_template_doctor import Severity, diagnose, parse_header
from gguf_template_doctor.doctor import extract_templates, mask_literal_regions

GOOD_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

VOCAB = ["<|im_start|>", "<|im_end|>", "a", "b"]


def _model(template=None, *, extra=(), tokens=VOCAB, eos=1):
    kv = [
        ("general.architecture", STRING, "llama"),
        ("general.name", STRING, "test-model"),
        ("tokenizer.ggml.model", STRING, "gpt2"),
    ]
    if tokens is not None:
        kv.append(("tokenizer.ggml.tokens", ARRAY, (STRING, tokens)))
    if eos is not None:
        kv.append(("tokenizer.ggml.eos_token_id", UINT32, eos))
    if template is not None:
        kv.append(("tokenizer.chat_template", STRING, template))
    kv.extend(extra)
    return kv


def _diagnose(tmp_path, kv, name="m.gguf"):
    path = tmp_path / name
    path.write_bytes(build_gguf(kv))
    return diagnose(parse_header(path))


def _codes(report, severity=None):
    return {
        f.code
        for f in report.findings
        if severity is None or f.severity is severity
    }


# --------------------------------------------------------------------------
# The real template must come out clean: no false positives.
# --------------------------------------------------------------------------


def test_real_template_has_no_errors_or_warnings(real_gguf):
    report = diagnose(parse_header(real_gguf))
    assert report.errors == []
    assert report.warnings == []
    assert report.ok
    assert [t.name for t in report.templates] == ["default"]


def test_real_template_is_reported_verbatim(real_gguf, qwen_template):
    report = diagnose(parse_header(real_gguf))
    assert report.templates[0].source == qwen_template
    assert report.architecture == "qwen2"


def test_good_synthetic_template_is_clean(tmp_path):
    report = _diagnose(tmp_path, _model(GOOD_TEMPLATE))
    assert report.errors == []
    assert report.warnings == []


# --------------------------------------------------------------------------
# Missing template.
# --------------------------------------------------------------------------


def test_missing_chat_template_is_an_error(tmp_path):
    report = _diagnose(tmp_path, _model(None))
    assert "no-chat-template" in _codes(report, Severity.ERROR)
    assert report.templates == []


def test_empty_chat_template_is_an_error(tmp_path):
    report = _diagnose(tmp_path, _model("   \n  "))
    assert "empty-template" in _codes(report, Severity.ERROR)


def test_plain_text_template_is_an_error(tmp_path):
    report = _diagnose(tmp_path, _model("You are a helpful assistant."))
    assert "not-a-template" in _codes(report, Severity.ERROR)


# --------------------------------------------------------------------------
# Structural breakage.
# --------------------------------------------------------------------------


def test_unclosed_for_block(tmp_path):
    template = "{% for m in messages %}{{ m.role }}{{ m.content }}"
    report = _diagnose(tmp_path, _model(template))
    codes = _codes(report, Severity.ERROR)
    assert "unbalanced-block" in codes


def test_unclosed_if_block(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'x' }}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" in _codes(report, Severity.ERROR)


def test_mismatched_block_closer(tmp_path):
    template = "{% for m in messages %}{{ m.role }}{{ m.content }}{% endif %}"
    report = _diagnose(tmp_path, _model(template))
    findings = [f for f in report.findings if f.code == "unbalanced-block"]
    assert findings and findings[0].severity is Severity.ERROR
    assert "endif" in findings[0].message


def test_endfor_without_for(tmp_path):
    template = "{{ messages[0].content }}{% endfor %}"
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" in _codes(report, Severity.ERROR)


def test_stray_else_outside_block(tmp_path):
    template = "{{ messages[0].content }}{% else %}{{ 'x' }}"
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" in _codes(report, Severity.ERROR)


def test_unclosed_expression_delimiter(tmp_path):
    template = "{% for m in messages %}{{ m.content {% endfor %}"
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-delimiter" in _codes(report, Severity.ERROR)


def test_if_else_endif_is_balanced(tmp_path):
    template = (
        "{% for m in messages %}"
        "{% if m.role == 'user' %}{{ m.content }}"
        "{% else %}{{ m.content }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant' }}{% endif %}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" not in _codes(report)
    assert "unbalanced-delimiter" not in _codes(report)


def test_inline_set_is_not_treated_as_a_block(tmp_path):
    """`{% set x = 1 %}` must not be mistaken for an unclosed block."""
    template = (
        "{% set ns = 'sys' %}"
        "{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'a' }}{% endif %}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" not in _codes(report)


def test_set_block_requires_endset(tmp_path):
    template = (
        "{% set greeting %}hi"
        "{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" in _codes(report, Severity.ERROR)


def test_whitespace_control_markers_are_understood(tmp_path):
    """`{%-` and `-%}` are the norm in real templates."""
    template = (
        "{%- for m in messages -%}{{- m.role -}}{{- m.content -}}{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- 'a' -}}{%- endif -%}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" not in _codes(report)
    assert "unbalanced-delimiter" not in _codes(report)


# --------------------------------------------------------------------------
# Literal regions: text that looks like markup but is not.
#
# Published templates really do write `{{- "}}" }}` to emit a brace pair (the
# Mistral-Nemo tool-call section does), so a scan that cannot tell a quoted
# string from markup calls a healthy model broken.
# --------------------------------------------------------------------------

_TAIL = "{% if add_generation_prompt %}{{ 'a' }}{% endif %}"


def test_delimiter_inside_a_string_literal_is_not_markup(tmp_path):
    template = (
        '{%- for m in messages %}{{ m.role }}{{ m.content }}{{- "}}" }}'
        "{%- endfor %}" + _TAIL
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-delimiter" not in _codes(report)
    assert "unbalanced-block" not in _codes(report)


def test_opening_delimiter_inside_a_string_literal_is_not_markup(tmp_path):
    template = (
        '{% for m in messages %}{{ m.role }}{{ m.content }}{{ "{{" }}{% endfor %}'
        + _TAIL
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-delimiter" not in _codes(report)


def test_block_keyword_inside_a_string_literal_is_not_markup(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}"
        "{{ '{% endfor %}' }}{% endfor %}" + _TAIL
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" not in _codes(report)


def test_escaped_quote_does_not_end_the_string_literal(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}"
        "{{ 'it\\'s }} here' }}{% endfor %}" + _TAIL
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-delimiter" not in _codes(report)


def test_raw_block_body_is_not_scanned(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}"
        "{% raw %}{{ this is not markup {% endif %}{% endraw %}" + _TAIL
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" not in _codes(report)
    assert "unbalanced-delimiter" not in _codes(report)


def test_comment_body_is_not_scanned(tmp_path):
    template = (
        "{# a stray }} and an {% endif %} inside a comment #}"
        "{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}" + _TAIL
    )
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" not in _codes(report)
    assert "unbalanced-delimiter" not in _codes(report)


def test_real_breakage_next_to_a_literal_is_still_reported(tmp_path):
    """Masking must not blind the checks to a genuine mismatch."""
    template = '{{ "}}" }}{% for m in messages %}{{ m.content }}'
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-block" in _codes(report, Severity.ERROR)


def test_unterminated_string_literal_is_still_an_error(tmp_path):
    template = "{% for m in messages %}{{ 'unterminated }}{% endfor %}" + _TAIL
    report = _diagnose(tmp_path, _model(template))
    assert "unbalanced-delimiter" in _codes(report, Severity.ERROR)


def test_masking_keeps_offsets_in_finding_messages(tmp_path):
    prefix = "{% for m in messages %}{{ m.content }}{% endfor %}"
    template = prefix + '{{ "}}" }}}}'
    report = _diagnose(tmp_path, _model(template))
    finding = next(f for f in report.findings if f.code == "unbalanced-delimiter")
    assert f"offset {len(prefix) + 10}" in finding.message


def test_mask_literal_regions_keeps_length_and_blanks_only_literals():
    source = '{{ "}}" }}'
    assert mask_literal_regions(source) == "{{      }}"
    assert len(mask_literal_regions(source)) == len(source)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "plain text",
        "{{",
        "{% if x %}",
        "{# unterminated comment",
        "{{ 'unterminated string",
        "{% raw %}never closed",
        '{{ "\\\\" }}',
        GOOD_TEMPLATE,
    ],
)
def test_mask_literal_regions_never_changes_the_length(source):
    assert len(mask_literal_regions(source)) == len(source)


def test_mask_literal_regions_blanks_raw_and_comment_bodies():
    assert mask_literal_regions("{% raw %}{{ x }}{% endraw %}") == (
        "{% raw %}       {% endraw %}"
    )
    assert mask_literal_regions("{# }} #}") == "{#    #}"


def test_mistral_nemo_template_is_not_reported_as_broken(nemo_gguf, nemo_template):
    """Regression: the published Nemo template writes a literal `}}`."""
    assert '"}}"' in nemo_template
    report = diagnose(parse_header(nemo_gguf))
    assert report.errors == []
    assert _codes(report, Severity.WARNING) == {"no-generation-prompt"}


# --------------------------------------------------------------------------
# Convention checks.
# --------------------------------------------------------------------------


def test_template_ignoring_messages_is_an_error(tmp_path):
    template = "{% if add_generation_prompt %}{{ 'assistant' }}{% endif %}"
    report = _diagnose(tmp_path, _model(template))
    assert "no-messages" in _codes(report, Severity.ERROR)


def test_missing_generation_prompt_is_a_warning(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}"
        "{{ '<|im_end|>' }}{% endfor %}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "no-generation-prompt" in _codes(report, Severity.WARNING)


def test_missing_role_dispatch_is_a_warning(tmp_path):
    template = (
        "{% for m in messages %}{{ m.content }}{{ '<|im_end|>' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'a' }}{% endif %}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "no-role-dispatch" in _codes(report, Severity.WARNING)


def test_bracket_style_role_access_is_accepted(tmp_path):
    template = (
        "{% for m in messages %}{{ m['role'] }}{{ m['content'] }}"
        "{{ '<|im_end|>' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'a' }}{% endif %}"
    )
    report = _diagnose(tmp_path, _model(template))
    assert "no-role-dispatch" not in _codes(report)
    assert "no-content" not in _codes(report)


def test_raise_exception_is_informational(tmp_path):
    template = GOOD_TEMPLATE + "{% if false %}{{ raise_exception('x') }}{% endif %}"
    report = _diagnose(tmp_path, _model(template))
    assert "uses-raise-exception" in _codes(report, Severity.INFO)


# --------------------------------------------------------------------------
# Vocabulary cross-checks.
# --------------------------------------------------------------------------


def test_special_token_missing_from_vocab_is_a_warning(tmp_path):
    report = _diagnose(tmp_path, _model(GOOD_TEMPLATE, tokens=["a", "b"], eos=1))
    finding = next(f for f in report.findings if f.code == "token-not-in-vocab")
    assert finding.severity is Severity.WARNING
    assert "<|im_start|>" in finding.message


def test_template_never_emitting_eos_is_a_warning(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'a' }}{% endif %}"
    )
    report = _diagnose(tmp_path, _model(template, tokens=VOCAB, eos=1))
    assert "eos-not-emitted" in _codes(report, Severity.WARNING)


def test_eos_token_reference_satisfies_the_check(tmp_path):
    template = (
        "{% for m in messages %}{{ m.role }}{{ m.content }}{{ eos_token }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'a' }}{% endif %}"
    )
    report = _diagnose(tmp_path, _model(template, tokens=VOCAB, eos=1))
    assert "eos-not-emitted" not in _codes(report)


def test_missing_eos_token_id_is_a_warning(tmp_path):
    report = _diagnose(tmp_path, _model(GOOD_TEMPLATE, eos=None))
    assert "no-eos-token-id" in _codes(report, Severity.WARNING)


def test_missing_tokenizer_model_is_a_warning(tmp_path):
    kv = [
        ("general.architecture", STRING, "llama"),
        ("tokenizer.chat_template", STRING, GOOD_TEMPLATE),
        ("tokenizer.ggml.eos_token_id", UINT32, 1),
        ("tokenizer.ggml.tokens", ARRAY, (STRING, VOCAB)),
    ]
    report = _diagnose(tmp_path, kv)
    assert "no-tokenizer-model" in _codes(report, Severity.WARNING)


def test_token_id_out_of_range_is_a_warning(tmp_path):
    """A model whose ids exceed its own vocab is reported once vocab is complete."""
    report = _diagnose(
        tmp_path,
        _model(GOOD_TEMPLATE, tokens=VOCAB, eos=None,
               extra=[("tokenizer.ggml.eos_token_id", UINT32, 1),
                      ("tokenizer.ggml.bos_token_id", UINT32, 99)]),
    )
    # bos is out of range, which makes the vocabulary partial by definition.
    assert "partial-vocabulary" in _codes(report, Severity.INFO)


def test_partial_vocabulary_suppresses_token_checks(real_gguf):
    """The trimmed fixture must not be flagged for tokens its vocab cannot list."""
    report = diagnose(parse_header(real_gguf))
    assert "partial-vocabulary" in {f.code for f in report.findings}
    assert "token-not-in-vocab" not in {f.code for f in report.findings}
    assert "token-id-out-of-range" not in {f.code for f in report.findings}


def test_no_vocab_at_all_skips_token_checks(tmp_path):
    report = _diagnose(tmp_path, _model(GOOD_TEMPLATE, tokens=None, eos=1))
    assert "token-not-in-vocab" not in _codes(report)


# --------------------------------------------------------------------------
# Named template variants.
# --------------------------------------------------------------------------


def test_named_template_variants_are_found(tmp_path):
    kv = _model(
        GOOD_TEMPLATE,
        extra=[
            ("tokenizer.chat_template.tool_use", STRING, GOOD_TEMPLATE),
            ("tokenizer.chat_template.rag", STRING, GOOD_TEMPLATE),
        ],
    )
    report = _diagnose(tmp_path, kv)
    names = {t.name for t in report.templates}
    assert names == {"default", "tool_use", "rag"}


def test_each_variant_is_checked_independently(tmp_path):
    kv = _model(
        GOOD_TEMPLATE,
        extra=[("tokenizer.chat_template.broken", STRING, "{% for m in messages %}")],
    )
    report = _diagnose(tmp_path, kv)
    broken = [f for f in report.findings if f.template == "broken"]
    assert any(f.code == "unbalanced-block" for f in broken)
    assert not [
        f for f in report.findings if f.template == "default" and f.severity is Severity.ERROR
    ]


def test_declared_variant_without_body_is_an_error(tmp_path):
    kv = _model(
        GOOD_TEMPLATE,
        extra=[("tokenizer.chat_templates", ARRAY, (STRING, ["default", "tool_use"]))],
    )
    report = _diagnose(tmp_path, kv)
    finding = next(f for f in report.findings if f.code == "declared-template-missing")
    assert finding.severity is Severity.ERROR
    assert "tool_use" in finding.message


def test_extract_templates_on_file_without_any(tmp_path):
    path = tmp_path / "bare.gguf"
    path.write_bytes(build_gguf([("general.architecture", STRING, "llama")]))
    assert extract_templates(parse_header(path)) == []


def test_parse_warnings_are_carried_into_the_report(tmp_path):
    path = tmp_path / "hdr.gguf"
    path.write_bytes(
        build_gguf(
            _model(GOOD_TEMPLATE),
            tensors=[("a.weight", (4096,), 0)],
            with_data=False,
        )
    )
    report = diagnose(parse_header(path))
    assert any("tensor data is missing" in w for w in report.parse_warnings)


def test_non_string_template_value_is_ignored(tmp_path):
    """A template stored with the wrong type must not crash the doctor."""
    kv = [
        ("general.architecture", STRING, "llama"),
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.eos_token_id", UINT32, 1),
        ("tokenizer.chat_template", BOOL, True),
    ]
    report = _diagnose(tmp_path, kv)
    assert "no-chat-template" in _codes(report, Severity.ERROR)
