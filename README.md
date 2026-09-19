# gguf-template-doctor

Offline CLI that reads a GGUF header, extracts the embedded chat template, and flags common
breakage without llama.cpp or network.

Only the header is read — magic, version, metadata key/value store and the tensor-info table.
Tensor payloads are never touched, so inspecting a 40 GB model costs the same as inspecting a
5 KB one. No third-party dependencies.

## Why this exists

A GGUF carries its chat template inside the file, and that copy goes stale: model authors ship
replacement `.jinja` files days after release, a "fixed" template turns out to emit a token the same
file's vocabulary never defines, and two quants of one model disagree about which version they
embed. Today that is diagnosed by running the model and noticing the output is wrong.

[`docs/motivation.md`](docs/motivation.md) records the four public reports this tool was built
against — link, date, author and quote for each — and states, per report, which check the tool runs
and what it finds. Each one has a matching test in `tests/test_scenarios.py` that builds a GGUF
carrying exactly that defect and asserts the tool names the cause.

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
| `--strict` | exit 1 on warnings too, not just errors (same as `--fail-on warning`) |
| `--fail-on {error,warning,info}` | lowest severity that makes the run exit 1 (default `error`) |

`--json` always emits strict JSON. GGUF float metadata may hold IEEE-754 NaN or infinities, which
have no JSON literal, so those values are emitted as the strings `"NaN"`, `"Infinity"` and
`"-Infinity"` rather than the bare tokens a strict parser would reject.

With `--list-metadata --json`, an array longer than 64 entries (a vocabulary is six figures long)
is reported as an object carrying its full length plus the first 64 entries, rather than in full,
so no key is ever dropped from the output:

```json
"tokenizer.ggml.tokens": { "truncated": true, "length": 151936, "items": ["!", "\"", "#", "..."] }
```

Arrays of 64 entries or fewer are emitted verbatim as plain JSON arrays.

The report is written with unencodable characters replaced, so a template containing non-ASCII text
still prints when stdout is opened in a narrow encoding (`LC_ALL=C`, a redirect under an ASCII
locale) instead of failing with a `UnicodeEncodeError`.

### Severity and exit codes

Findings carry one of three severities, and the severity decides whether the run fails:

| Severity | Means | Exit code by default |
| --- | --- | --- |
| `error` | the template is structurally broken or absent; a client cannot use it as-is | `1` |
| `warning` | likely wrong but still usable — the model loads and generates | `0` |
| `info` | context for reading the rest of the report, not a defect | `0` |

**Warnings do not fail the run by default.** A warning describes a file that works; `no-generation-prompt`
in particular fires on healthy, widely used models — the whole Mistral `[INST]` family formats the
assistant turn without ever consulting `add_generation_prompt`. Failing on warnings would report
those models as broken, so the default threshold is `error`.

Pass `--strict` (equivalently `--fail-on warning`) when you do want warnings to fail, for example in
a CI step gating a model release. `--fail-on info` additionally fails on `info` findings. When
`--strict` and `--fail-on` disagree, the stricter of the two applies.

```sh
$ gguf-template-doctor mistral-nemo.gguf        # 1 warning
summary: 0 error(s), 1 warning(s), 2 info
note: exit code reflects errors only (1 warning(s) ignored); use --strict to fail on warnings too
$ echo $?
0
$ gguf-template-doctor mistral-nemo.gguf --strict; echo $?
1
```

Every finding is printed regardless of the threshold — the flag changes the exit code, never the
report. With `--json`, the payload carries `fail_on` (the threshold in force) and `exit_code`
alongside `ok`, which keeps its own meaning: `ok` describes the file (nothing worse than `info`),
`exit_code` describes what this invocation decided about it.

| Code | Meaning |
| --- | --- |
| `0` | the file was read and nothing at or above the failure threshold was found |
| `1` | the file was read and the doctor reported a finding at that threshold |
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

Structural problems (reported as `error`, exit 1):

- `no-chat-template` — `tokenizer.chat_template` is absent, so a client has to guess the format
- `empty-template` / `not-a-template` — the template is blank, or contains no Jinja at all
- `unbalanced-delimiter` — an unclosed `{{`, `{%` or `{#`
- `unbalanced-block` — `{% if %}` / `{% for %}` blocks that do not match their closers
- `no-messages` — the template never references `messages`, so the conversation is dropped
- `declared-template-missing` — `tokenizer.chat_templates` names a variant that is not present

Likely-wrong-but-usable problems (reported as `warning`, exit 0 unless `--strict`):

- `no-generation-prompt` — never checks `add_generation_prompt`, so generation may not start
- `no-role-dispatch` / `no-content` — never reads a message `role` / `content`
- `token-not-in-vocab` — the template emits a special token that is not in `tokenizer.ggml.tokens`
- `eos-not-emitted` — the template never emits EOS nor references `eos_token`
- `no-tokenizer-model` / `no-eos-token-id` — tokenizer metadata gaps
- `token-id-out-of-range` — `eos`/`bos`/`padding` token id points past the model's vocabulary

The checks are structural rather than a full Jinja2 evaluation: the tool has no dependencies and
runs offline, and the failures that actually break deployments are structural. Literal regions of
the template (quoted strings, comments, `{% raw %}` bodies) are excluded from the structural scans,
so text that merely looks like markup is not counted as markup. Severities are tiered, and only
`error` fails the run by default, so a legitimate production template — which may use constructs a
linter cannot fully evaluate — is not reported as broken. Named variants
(`tokenizer.chat_template.tool_use` and friends) are each checked independently.

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

Closing delimiters in ordinary body text (`}}`, `%}`, `#}` outside any tag) are output, not markup,
and are not reported; Jinja's `{%+` whitespace-control marker is understood like `{%-`.

A tensor whose `ggml` type this tool does not know has an unknown element size, so the size of the
tensor data section cannot be computed. The report says so rather than guessing a size — the
missing-tensor-data check is skipped for such a file instead of producing a fabricated number.

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

`tests/test_scenarios.py` holds one test per public report in
[`docs/motivation.md`](docs/motivation.md). Each builds a GGUF carrying exactly the template defect
its report describes, runs the CLI over it, and asserts the tool both finds the defect and names the
cause; the source link and quote are in the test's docstring.

## Releases

Pushing a `vX.Y.Z` tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml),
which runs the test suite, checks the tag against the packaged version, builds a wheel and an
sdist and attaches both to the GitHub Release.

The version is declared in two places — `version` in `pyproject.toml` (what goes into the wheel
metadata) and `__version__` in `src/gguf_template_doctor/__init__.py` (what
`gguf-template-doctor --version` prints). Both must equal the tag, or the release fails before
anything is built; `scripts/check_version_tag.py` is the check, and it can be run locally:

```sh
python scripts/check_version_tag.py v0.1.0
```

Nothing is published to PyPI.
