"""Cross-check the structural checks against a real Jinja parser.

The package itself has no dependencies, so Jinja2 is a development extra and
this module skips when it is not installed.  Its job is to keep the lexical
checks from drifting away from what a Jinja parser accepts: a template the
parser is happy with must never be reported as structurally broken, and the
breakage the parser rejects must still be reported.
"""

from __future__ import annotations

import pytest

jinja2 = pytest.importorskip("jinja2")

from gguf_template_doctor import parse_header  # noqa: E402
from gguf_template_doctor.doctor import check_template_structure  # noqa: E402

STRUCTURAL_CODES = {"unbalanced-block", "unbalanced-delimiter"}

VALID = [
    "hello",
    "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}",
    "{%- if x -%}{{ x }}{%- else -%}{{ 'n' }}{%- endif -%}",
    '{{- "}}" }}',
    '{{ "{{" }}',
    "{{ '{% endfor %}' }}",
    "{{ 'it\\'s }} here' }}",
    "{% raw %}{{ this is not markup {% endif %}{% endraw %}",
    "{# a stray }} and an {% endif %} in a comment #}",
    "{% set x = 1 %}{{ x }}",
    "{% set body %}text{% endset %}{{ body }}",
    "{% macro render(m) %}{{ m }}{% endmacro %}{{ render('a') }}",
    "{% filter upper %}shout{% endfilter %}",
    "{% for m in messages %}{{ m.content }}{% if loop.last %}{{ '!' }}{% endif %}"
    "{% endfor %}",
]

INVALID = [
    "{% for m in messages %}{{ m.content }}",
    "{{ m.content }}{% endfor %}",
    "{% if x %}{{ a }}{% endfor %}",
    "{{ unclosed expression",
    "{% raw %}{{ x }}",
    "{% set body %}text",
]


def _structural_codes(source: str) -> set[str]:
    return {
        f.code for f in check_template_structure(source) if f.code in STRUCTURAL_CODES
    }


def _jinja_accepts(source: str) -> bool:
    try:
        jinja2.Environment().parse(source)
    except jinja2.TemplateSyntaxError:
        return False
    return True


@pytest.mark.parametrize("source", VALID)
def test_template_jinja_accepts_is_not_called_unbalanced(source):
    assert _jinja_accepts(source), "test data is wrong: Jinja rejects this"
    assert _structural_codes(source) == set()


@pytest.mark.parametrize("source", INVALID)
def test_structural_breakage_jinja_rejects_is_reported(source):
    assert not _jinja_accepts(source), "test data is wrong: Jinja accepts this"
    assert _structural_codes(source)


def test_published_templates_agree_with_jinja(real_gguf, nemo_gguf):
    for path in (real_gguf, nemo_gguf):
        source = parse_header(path).metadata["tokenizer.chat_template"]
        assert _jinja_accepts(source)
        assert _structural_codes(source) == set()
