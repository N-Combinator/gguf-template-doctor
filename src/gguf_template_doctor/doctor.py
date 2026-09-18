"""Checks applied to a chat template extracted from GGUF metadata.

The checks are deliberately structural rather than a full Jinja2 implementation:
the tool must run offline with no dependencies, and the failures that actually
break llama.cpp deployments are structural (unbalanced blocks, a template that
never emits a generation prompt, special tokens that are not in the vocabulary).

Every finding carries a severity so that a real production template - which may
legitimately use constructs a linter cannot fully evaluate - does not get flagged
as broken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .reader import GgufFile

#: Metadata keys that have carried a chat template, in priority order.
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
#: Multi-template models expose named variants as tokenizer.chat_template.<name>
#: alongside a tokenizer.chat_templates list of names.
CHAT_TEMPLATE_NAMES_KEY = "tokenizer.chat_templates"
CHAT_TEMPLATE_PREFIX = "tokenizer.chat_template."


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """One diagnosed problem (or observation) about a template."""

    severity: Severity
    code: str
    message: str
    #: Which template this concerns: "default" or a variant name.
    template: str = "default"

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "template": self.template,
        }


@dataclass
class ChatTemplate:
    """A chat template found in metadata."""

    name: str
    source: str
    key: str


@dataclass
class Report:
    """Result of examining one GGUF file."""

    path: str
    architecture: str | None = None
    model_name: str | None = None
    templates: list[ChatTemplate] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    #: Warnings produced by the header parser itself.
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing worse than an INFO finding was reported."""
        return not self.errors and not self.warnings


# Jinja block openers and the closer each one requires.
_BLOCK_PAIRS = {
    "if": "endif",
    "for": "endfor",
    "macro": "endmacro",
    "call": "endcall",
    "filter": "endfilter",
    "raw": "endraw",
    "with": "endwith",
    "block": "endblock",
    "set": "endset",  # only when used as a block, handled below
}
_CLOSERS = {v: k for k, v in _BLOCK_PAIRS.items()}
_MID_BLOCK = {"else", "elif"}

_STATEMENT_RE = re.compile(r"\{%-?\s*(\w+)")
_TOKEN_RE = re.compile(r"<\|[^|>]{1,64}\|>|\[/?(?:INST|SYS)\]|<s>|</s>")

_TAG_OPEN_RE = re.compile(r"\{[{%#]")
_RAW_TAG_RE = re.compile(r"\{%-?\s*raw\s*-?%\}")
_ENDRAW_TAG_RE = re.compile(r"\{%-?\s*endraw\s*-?%\}")


def _blank(out: list[str], start: int, stop: int) -> None:
    """Replace ``out[start:stop]`` with spaces, keeping the length identical."""
    for index in range(start, min(stop, len(out))):
        if out[index] != "\n":
            out[index] = " "


def _blank_string(out: list[str], source: str, index: int) -> int:
    """Blank the quoted string starting at *index*; return the offset after it."""
    quote = source[index]
    length = len(source)
    _blank(out, index, index + 1)
    position = index + 1
    while position < length:
        char = source[position]
        if char == "\\" and position + 1 < length:
            _blank(out, position, position + 2)
            position += 2
            continue
        _blank(out, position, position + 1)
        position += 1
        if char == quote:
            break
    return position


def mask_literal_regions(source: str) -> str:
    """Return *source* with its literal regions replaced by spaces.

    The block and delimiter checks are lexical: without this, a `}}` written
    inside a quoted Jinja string - which published templates really do, to emit
    a literal brace - is counted as markup and reported as a broken template.
    Blanked are the contents of quoted strings inside `{{ }}` / `{% %}` tags,
    the bodies of `{# #}` comments and the bodies of `{% raw %}` blocks.

    The result has exactly the same length as *source*, and newlines are kept,
    so offsets reported by the checks still point into the original template.
    """
    out = list(source)
    length = len(source)
    index = 0
    while index < length:
        match = _TAG_OPEN_RE.search(source, index)
        if match is None:
            break
        start = match.start()
        kind = source[start + 1]
        if kind == "#":
            end = source.find("#}", start + 2)
            stop = length if end == -1 else end
            _blank(out, start + 2, stop)
            index = length if end == -1 else end + 2
            continue
        closer = "}}" if kind == "{" else "%}"
        position = start + 2
        while position < length:
            char = source[position]
            if char in "\"'":
                position = _blank_string(out, source, position)
                continue
            if source.startswith(closer, position):
                position += 2
                break
            position += 1
        if kind == "%" and _RAW_TAG_RE.match(source, start):
            end_match = _ENDRAW_TAG_RE.search(source, position)
            stop = length if end_match is None else end_match.start()
            _blank(out, position, stop)
            position = stop
        index = max(position, start + 1)
    return "".join(out)


def extract_templates(gguf: GgufFile) -> list[ChatTemplate]:
    """Collect the default template and any named variants from metadata."""
    templates: list[ChatTemplate] = []
    default = gguf.metadata.get(CHAT_TEMPLATE_KEY)
    if isinstance(default, str):
        templates.append(ChatTemplate("default", default, CHAT_TEMPLATE_KEY))
    for key, value in gguf.metadata.items():
        if key.startswith(CHAT_TEMPLATE_PREFIX) and isinstance(value, str):
            name = key[len(CHAT_TEMPLATE_PREFIX) :]
            templates.append(ChatTemplate(name, value, key))
    return templates


def _check_balanced_blocks(source: str, name: str) -> list[Finding]:
    """Match {% if %}/{% endif %} style blocks, reporting the first mismatch."""
    findings: list[Finding] = []
    stack: list[str] = []
    for match in _STATEMENT_RE.finditer(source):
        keyword = match.group(1)
        if keyword == "set":
            # `{% set x = 1 %}` is a statement; `{% set x %}...{% endset %}` is a
            # block.  Only the form without `=` opens a block.
            statement_end = source.find("%}", match.end())
            body = source[match.end() : statement_end if statement_end != -1 else None]
            if "=" in body:
                continue
            stack.append("set")
            continue
        if keyword in _BLOCK_PAIRS:
            stack.append(keyword)
        elif keyword in _CLOSERS:
            expected = _CLOSERS[keyword]
            if not stack:
                findings.append(
                    Finding(
                        Severity.ERROR,
                        "unbalanced-block",
                        f"{{% {keyword} %}} at offset {match.start()} closes a "
                        f"{{% {expected} %}} block that was never opened",
                        name,
                    )
                )
                return findings
            opened = stack.pop()
            if opened != expected:
                findings.append(
                    Finding(
                        Severity.ERROR,
                        "unbalanced-block",
                        f"{{% {keyword} %}} at offset {match.start()} closes a "
                        f"{{% {opened} %}} block (expected "
                        f"{{% {_BLOCK_PAIRS[opened]} %}})",
                        name,
                    )
                )
                return findings
        elif keyword in _MID_BLOCK and not stack:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "unbalanced-block",
                    f"{{% {keyword} %}} at offset {match.start()} appears outside "
                    "any block",
                    name,
                )
            )
            return findings
    if stack:
        unclosed = stack[-1]
        findings.append(
            Finding(
                Severity.ERROR,
                "unbalanced-block",
                f"{{% {unclosed} %}} block is never closed "
                f"(missing {{% {_BLOCK_PAIRS[unclosed]} %}})",
                name,
            )
        )
    return findings


def _check_delimiters(source: str, name: str) -> list[Finding]:
    """Look for delimiters that are opened and never closed."""
    findings: list[Finding] = []
    for opener, closer, label in (
        ("{%", "%}", "statement"),
        ("{{", "}}", "expression"),
        ("{#", "#}", "comment"),
    ):
        depth = 0
        index = 0
        while index < len(source):
            next_open = source.find(opener, index)
            next_close = source.find(closer, index)
            if next_open == -1 and next_close == -1:
                break
            if next_open != -1 and (next_close == -1 or next_open < next_close):
                depth += 1
                index = next_open + len(opener)
            else:
                if depth == 0:
                    findings.append(
                        Finding(
                            Severity.ERROR,
                            "unbalanced-delimiter",
                            f"stray {closer!r} at offset {next_close} with no "
                            f"matching {opener!r} ({label})",
                            name,
                        )
                    )
                    return findings
                depth -= 1
                index = next_close + len(closer)
        if depth:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "unbalanced-delimiter",
                    f"{depth} unclosed {opener!r} {label} delimiter(s); "
                    f"missing {closer!r}",
                    name,
                )
            )
    return findings


def check_template_structure(source: str, name: str = "default") -> list[Finding]:
    """Run the lexical structure checks on one template.

    Literal regions are blanked first (see :func:`mask_literal_regions`), so a
    delimiter or block keyword written inside a string, a comment or a
    `{% raw %}` body is not counted as markup.
    """
    masked = mask_literal_regions(source)
    return _check_delimiters(masked, name) + _check_balanced_blocks(masked, name)


def _check_conventions(source: str, name: str) -> list[Finding]:
    """Check the things a chat template is expected to do."""
    findings: list[Finding] = []
    if not source.strip():
        findings.append(
            Finding(Severity.ERROR, "empty-template", "template is empty", name)
        )
        return findings
    if "{{" not in source and "{%" not in source:
        findings.append(
            Finding(
                Severity.ERROR,
                "not-a-template",
                "template contains no Jinja expressions or statements; it would "
                "render the same text for every conversation",
                name,
            )
        )
        return findings
    if "messages" not in source:
        findings.append(
            Finding(
                Severity.ERROR,
                "no-messages",
                "template never references `messages`, so the conversation would "
                "be dropped",
                name,
            )
        )
    if "add_generation_prompt" not in source:
        findings.append(
            Finding(
                Severity.WARNING,
                "no-generation-prompt",
                "template never checks `add_generation_prompt`; inference servers "
                "use it to append the assistant turn header, and without it "
                "generation may not start",
                name,
            )
        )
    if not re.search(r"\.role\b|\[['\"]role['\"]\]", source):
        findings.append(
            Finding(
                Severity.WARNING,
                "no-role-dispatch",
                "template never reads a message `role`; user and assistant turns "
                "would be formatted identically",
                name,
            )
        )
    if not re.search(r"\.content\b|\[['\"]content['\"]\]", source):
        findings.append(
            Finding(
                Severity.WARNING,
                "no-content",
                "template never reads a message `content`",
                name,
            )
        )
    if re.search(r"\braise_exception\b", source):
        findings.append(
            Finding(
                Severity.INFO,
                "uses-raise-exception",
                "template calls `raise_exception`; some minimal Jinja runtimes do "
                "not provide it",
                name,
            )
        )
    return findings


def declared_vocabulary_size(gguf: GgufFile) -> int | None:
    """Vocabulary size the file declares, judged without looking at token ids.

    Two independent sources are consulted, in order of directness:

    * ``<arch>.vocab_size`` when the architecture publishes it;
    * the vocabulary dimension of the token-embedding tensor - GGUF stores
      ``token_embd.weight`` (and the untied ``output.weight``) as
      ``[n_embd, n_vocab]``, so its second dimension is the real vocabulary size.

    Neither depends on ``tokenizer.ggml.*_token_id`` being correct, which is what
    lets a trimmed vocabulary be told apart from a genuinely wrong token id.
    Returns ``None`` when the file declares its vocabulary size nowhere.
    """
    architecture = gguf.metadata.get("general.architecture")
    if isinstance(architecture, str):
        size = gguf.metadata.get(f"{architecture}.vocab_size")
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            return size
    tensors = {tensor.name: tensor for tensor in gguf.tensors}
    for name in ("token_embd.weight", "output.weight"):
        tensor = tensors.get(name)
        if tensor is None or len(tensor.dimensions) != 2:
            continue
        if tensor.dimensions[1] > 0:
            return tensor.dimensions[1]
    return None


def vocabulary_shortfall(gguf: GgufFile) -> tuple[int, int] | None:
    """``(present, declared)`` when the token list is shorter than declared.

    A file can carry fewer entries in ``tokenizer.ggml.tokens`` than the
    vocabulary it declares: slimmed fixtures do it on purpose, and tools that
    rewrite metadata have been known to drop the vocab.  Template checks that
    look tokens up are meaningless then, because every token looks absent.
    Returns ``None`` when the vocabulary is complete or its size is undeclared.
    """
    tokens = gguf.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list):
        return None
    declared = declared_vocabulary_size(gguf)
    if declared is None or len(tokens) >= declared:
        return None
    return len(tokens), declared


def _vocabulary(gguf: GgufFile) -> set[str] | None:
    tokens = gguf.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list):
        return None
    return {token for token in tokens if isinstance(token, str)}


def _check_special_tokens(
    source: str, name: str, vocabulary: set[str] | None
) -> list[Finding]:
    """Verify special tokens used by the template exist in the vocabulary.

    Skipped when the vocabulary is absent or was trimmed, because a partial
    vocabulary would make every token look missing.
    """
    findings: list[Finding] = []
    if not vocabulary:
        return findings
    used = {m.group(0) for m in _TOKEN_RE.finditer(source)}
    missing = sorted(token for token in used if token not in vocabulary)
    if missing:
        findings.append(
            Finding(
                Severity.WARNING,
                "token-not-in-vocab",
                "template uses special token(s) that are not in "
                f"tokenizer.ggml.tokens: {', '.join(missing)}",
                name,
            )
        )
    return findings


def _eos_token(gguf: GgufFile) -> str | None:
    token_id = gguf.metadata.get("tokenizer.ggml.eos_token_id")
    tokens = gguf.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(token_id, int) or not isinstance(tokens, list):
        return None
    if 0 <= token_id < len(tokens) and isinstance(tokens[token_id], str):
        return tokens[token_id]
    return None


def _check_metadata(gguf: GgufFile) -> list[Finding]:
    """Checks on the tokenizer metadata the template depends on."""
    findings: list[Finding] = []
    if "tokenizer.ggml.model" not in gguf.metadata:
        findings.append(
            Finding(
                Severity.WARNING,
                "no-tokenizer-model",
                "tokenizer.ggml.model is missing; the file may not be usable for "
                "inference",
                "-",
            )
        )
    if "tokenizer.ggml.eos_token_id" not in gguf.metadata:
        findings.append(
            Finding(
                Severity.WARNING,
                "no-eos-token-id",
                "tokenizer.ggml.eos_token_id is missing; generation may not stop",
                "-",
            )
        )
    tokens = gguf.metadata.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list):
        # An id is out of range when it exceeds the vocabulary the model really
        # has, which is the declared size when this file carries only part of the
        # token list.  A trimmed vocab therefore stays silent while a genuinely
        # wrong id - the classic bad-conversion failure that breaks stop-token
        # handling - is reported whether the vocab is complete or not.
        shortfall = vocabulary_shortfall(gguf)
        if shortfall is None:
            limit, source = len(tokens), "the vocabulary in this file has"
        else:
            limit, source = shortfall[1], "this file declares a vocabulary of"
        for key in (
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.padding_token_id",
        ):
            value = gguf.metadata.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            if value >= limit:
                findings.append(
                    Finding(
                        Severity.WARNING,
                        "token-id-out-of-range",
                        f"{key} is {value} but {source} only {limit} entries",
                        "-",
                    )
                )
    return findings


def diagnose(gguf: GgufFile) -> Report:
    """Run every check against *gguf* and return a :class:`Report`."""
    report = Report(
        path=str(gguf.path),
        architecture=(
            gguf.metadata.get("general.architecture")
            if isinstance(gguf.metadata.get("general.architecture"), str)
            else None
        ),
        model_name=(
            gguf.metadata.get("general.name")
            if isinstance(gguf.metadata.get("general.name"), str)
            else None
        ),
        parse_warnings=list(gguf.warnings),
    )
    report.templates = extract_templates(gguf)

    if not report.templates:
        report.findings.append(
            Finding(
                Severity.ERROR,
                "no-chat-template",
                f"no chat template in metadata: {CHAT_TEMPLATE_KEY} is absent, so "
                "a client has to guess the prompt format",
                "-",
            )
        )
        report.findings.extend(_check_metadata(gguf))
        return report

    shortfall = vocabulary_shortfall(gguf)
    # A trimmed vocabulary would make every special token look absent, so the
    # vocabulary-dependent checks are skipped and the reason is reported instead.
    # Token-id range checking is not skipped: it lives in _check_metadata and
    # compares against the declared size rather than the truncated token list.
    vocabulary = None if shortfall else _vocabulary(gguf)
    eos = None if shortfall else _eos_token(gguf)
    if shortfall:
        present, declared = shortfall
        report.findings.append(
            Finding(
                Severity.INFO,
                "partial-vocabulary",
                f"tokenizer.ggml.tokens holds {present} entries but this file "
                f"declares a vocabulary of {declared}, so the vocabulary is "
                "incomplete; vocabulary-dependent template checks were skipped",
                "-",
            )
        )
    for template in report.templates:
        source = template.source
        name = template.name
        report.findings.extend(check_template_structure(source, name))
        report.findings.extend(_check_conventions(source, name))
        report.findings.extend(_check_special_tokens(source, name, vocabulary))
        if eos and eos not in source and "eos_token" not in source:
            report.findings.append(
                Finding(
                    Severity.WARNING,
                    "eos-not-emitted",
                    f"template never emits the EOS token ({eos!r}) or references "
                    "`eos_token`; turns may not be terminated",
                    name,
                )
            )

    names = gguf.metadata.get(CHAT_TEMPLATE_NAMES_KEY)
    if isinstance(names, list):
        declared = {n for n in names if isinstance(n, str)}
        present = {t.name for t in report.templates}
        for missing in sorted(declared - present):
            report.findings.append(
                Finding(
                    Severity.ERROR,
                    "declared-template-missing",
                    f"{CHAT_TEMPLATE_NAMES_KEY} lists {missing!r} but "
                    f"{CHAT_TEMPLATE_PREFIX}{missing} is not in metadata",
                    missing,
                )
            )

    report.findings.extend(_check_metadata(gguf))
    return report


def summarize(report: Report) -> dict[str, Any]:
    """JSON-serialisable form of a report."""
    return {
        "path": report.path,
        "architecture": report.architecture,
        "model_name": report.model_name,
        "templates": [
            {"name": t.name, "key": t.key, "length": len(t.source)}
            for t in report.templates
        ],
        "parse_warnings": report.parse_warnings,
        "findings": [f.as_dict() for f in report.findings],
        "ok": report.ok,
    }
