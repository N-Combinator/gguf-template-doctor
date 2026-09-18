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
- `no-tokenizer-model` / `no-eos-token-id` / `token-id-out-of-range` — tokenizer metadata gaps

The checks are structural rather than a full Jinja2 evaluation: the tool has no dependencies and
runs offline, and the failures that actually break deployments are structural. Severities are
tiered so that a legitimate production template — which may use constructs a linter cannot fully
evaluate — is not reported as broken. Named variants (`tokenizer.chat_template.tool_use` and
friends) are each checked independently.

Vocabulary-dependent checks are skipped when the file's own special-token ids point past the end
of `tokenizer.ggml.tokens`, since an incomplete vocabulary would make every token look missing.

## Supported GGUF

GGUF versions 2 and 3, all 13 metadata value types from the specification (`uint8`…`float64`,
`bool`, `string`), arrays of any of them, and nested arrays. Version 1 used 32-bit counts and is
rejected explicitly rather than mis-parsed.

## Limitations

Known limitation of v0.1: the lexical block check does not distinguish literal regions of a Jinja
template — text inside `{% raw %}` … `{% endraw %}`, or delimiters written inside literal strings —
from real markup, so `unbalanced-block` and `unbalanced-delimiter` can produce a false positive or
miss a genuine mismatch in those places. This is an accepted limitation of version 0.1.

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
tensor table shortened to 3 entries so the file can live in the repository. It records its own
provenance in `general.source.url` and `testing.vocab_trimmed_from`. The suite keeps running
fully offline.
