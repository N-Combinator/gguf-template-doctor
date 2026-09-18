# gguf-template-doctor

Offline CLI that reads a GGUF header, extracts the embedded chat template, and flags common
breakage without llama.cpp or network.

Only the header is read — magic, version, metadata key/value store and the tensor-info table.
Tensor payloads are never touched, so inspecting a 40 GB model costs the same as inspecting a
5 KB one. No third-party dependencies.

## Install

```sh
pip install -e .
```

## Usage

```sh
gguf-template-doctor model.gguf
```

```
file:         model.gguf
gguf version: 3
architecture: qwen2
model:        qwen2.5-0.5b-instruct
tensors:      291
metadata:     26 keys
chat template(s): default
  default: 2509 chars from tokenizer.chat_template

findings: none
summary: 0 error(s), 0 warning(s), 0 info
```

| Flag | Effect |
| --- | --- |
| `--json` | emit the report as JSON instead of text |
| `--show-template` | print the chat template source |
| `--list-metadata` | print every metadata key with its type and value |
| `--strict` | exit 1 on `info` findings too, not just warnings and errors |

`--json` always emits strict JSON. GGUF float metadata may hold IEEE-754 NaN or infinities, which
have no JSON literal, so those values are emitted as the strings `"NaN"`, `"Infinity"` and
`"-Infinity"` rather than the bare tokens a strict parser would reject.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | the file was read and nothing worse than an `info` finding was found |
| `1` | the file was read but the doctor reported warnings or errors |
| `2` | the file could not be read: missing, not GGUF, truncated or malformed |

A file that cannot be parsed always produces a single-line diagnosis on stderr and exit code 2,
never a traceback:

```sh
$ gguf-template-doctor truncated.gguf
gguf-template-doctor: error: file ends inside value of metadata key 'tokenizer.chat_template': wanted 2509 bytes at offset 2732 but the file is only 3000 bytes
$ echo $?
2
```

## What the doctor checks

Structural problems (reported as `error`):

- `no-chat-template` — `tokenizer.chat_template` is absent, so a client has to guess the format
- `empty-template` / `not-a-template` — the template is blank, or contains no Jinja at all
- `unbalanced-delimiter` — an unclosed `{{`, `{%` or `{#`
- `unbalanced-block` — `{% if %}` / `{% for %}` blocks that do not match their closers
- `no-messages` — the template never references `messages`, so the conversation is dropped
- `declared-template-missing` — `tokenizer.chat_templates` names a variant that is not present

Likely-wrong-but-usable problems (reported as `warning`):

- `no-generation-prompt` — never checks `add_generation_prompt`, so generation may not start
- `no-role-dispatch` / `no-content` — never reads a message `role` / `content`
- `token-not-in-vocab` — the template emits a special token that is not in `tokenizer.ggml.tokens`
- `eos-not-emitted` — the template never emits EOS nor references `eos_token`
- `no-tokenizer-model` / `no-eos-token-id` — tokenizer metadata gaps
- `token-id-out-of-range` — `eos`/`bos`/`padding` token id points past the model's vocabulary

The checks are structural rather than a full Jinja2 evaluation: the tool has no dependencies and
runs offline, and the failures that actually break deployments are structural. Literal regions of
the template (quoted strings, comments, `{% raw %}` bodies) are excluded from the structural scans,
so text that merely looks like markup is not counted as markup. Severities are
tiered so that a legitimate production template — which may use constructs a linter cannot fully
evaluate — is not reported as broken. Named variants (`tokenizer.chat_template.tool_use` and
friends) are each checked independently.

Some files carry only part of their vocabulary — trimmed test fixtures, or metadata rewritten by a
tool that dropped the token list. That is detected from evidence independent of the token ids: the
declared `<arch>.vocab_size`, or the vocabulary dimension of `token_embd.weight`. When
`tokenizer.ggml.tokens` is shorter than the vocabulary the file declares, `partial-vocabulary`
(`info`) is reported and the checks that look tokens up (`token-not-in-vocab`, `eos-not-emitted`)
are skipped, since an incomplete token list would make every token look missing. Token ids are
still range-checked in that case — against the declared size — so an id that no vocabulary of this
model could hold is reported either way.

## Supported GGUF

GGUF versions 2 and 3, all 13 metadata value types from the specification (`uint8`…`float64`,
`bool`, `string`), arrays of any of them, and nested arrays. Version 1 used 32-bit counts and is
rejected explicitly rather than mis-parsed.

## Limitations

The structural checks are lexical, not a Jinja parser. Before scanning, literal regions are blanked
out — the contents of quoted strings inside `{{ … }}` / `{% … %}`, the bodies of `{# … #}` comments
and the bodies of `{% raw %}` … `{% endraw %}` blocks — so a template that writes `{{- "}}" }}` to
emit a literal brace pair is not reported as broken. Offsets in findings still refer to the original
template.

What remains out of reach is everything below the level of delimiters and block keywords: an
expression that references an undefined variable, a filter that does not exist, a wrong argument
count, `{% if %}` conditions that can never be true. `unbalanced-block` and `unbalanced-delimiter`
answer the question "is this template structurally well-formed", not "does it render correctly".

## Not implemented

`--compare-known` (comparing a template against known-good reference templates) is deliberately
absent: it needs an agreed source of reference templates, and none was specified. Adding it would
mean inventing the ground truth, which is worse than not having the flag.

## Tests

```sh
pip install -e ".[dev]"
pytest
```

`tests/fixtures/qwen2.5-0.5b-instruct-header.gguf` is the header of a published model
([Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), q2_k) so
the parser is exercised against a real file and not only against synthetic fixtures. Every
metadata value in it is the real one byte for byte, including the 2509-character chat template;
only the three large vocabulary arrays (151936 entries) were truncated to 64 entries and the
tensor table shortened to 3 entries so the file can live in the repository. Those 3 tensor-info
records are the published ones, so `token_embd.weight` keeps its real `[896, 151936]` shape and the
file still states the vocabulary size it was built with. It records its own provenance in
`general.source.url` and `testing.vocab_trimmed_from`. The suite keeps running fully offline.

`tests/fixtures/mistral-nemo-instruct-2407-header.gguf` is a second such header, from
[bartowski/Mistral-Nemo-Instruct-2407-GGUF](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF)
(IQ2_M), trimmed the same way (131072 vocabulary entries → 64). It is kept as a regression fixture:
its 3945-character template emits a literal `}}` from inside a quoted string, which an earlier
version of the delimiter scan reported as an error on a healthy published model.

`tests/test_jinja_agreement.py` cross-checks the structural verdicts against a real Jinja parser —
anything Jinja2 accepts must not be reported as unbalanced, and structural breakage it rejects must
be. Jinja2 is a test-only extra; it is never imported by the package, and the test skips if it is
not installed.
