# Why this tool exists

Four public reports, all from r/LocalLLaMA in 2026, describe the same class of problem: the chat
template baked into a GGUF file is broken, stale, or disagrees with the file's own vocabulary, and
the only way to find out today is to run the model and notice the output is wrong. Each section
below records one of those reports and the single thing this tool does about it.

The links, dates, subreddit, authors and quotes are copied verbatim from the evidence table that
motivated the issue; nothing here is paraphrased or reconstructed.

A note on scope, stated once. This tool reports what a file *is*: it reads the embedded template
and cross-checks it against the file's own metadata. It cannot say whether a template matches an
upstream "fixed" version, because that requires a reference table of known-good templates and no
source for one was specified (see `--compare-known` under "Not implemented" in the README). Where a
report below is really a question about upstream, the section says so and describes the narrower
thing the tool actually answers.

---

## 1. A model family ships replacement templates because the embedded one is broken

- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1shs6sx/more_gemma4_fixes_in_the_past_24_hours/
- **Date:** 2026-04-10
- **Subreddit:** r/LocalLLaMA
- **Author:** andy2na
- **Quote:** "New chat templates from Google to fix tool calling" (linking a new .jinja file to replace the broken embedded one)

**What the tool does with this:** it parses the embedded template's block structure and reports
`unbalanced-block` (an `error`, exit 1) naming the offending tag and its offset — so a tool-calling
branch whose `{% if %}` is closed by the wrong tag is found by reading the file, not by watching
tool calls fail at runtime.

Test: `test_gemma_tool_calling_branch_is_structurally_broken`.

---

## 2. "Is the fix already in the GGUF I downloaded?"

- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1v0pp3s/qwen_36_27b_opencode_what_am_i_doing_wrong/
- **Date:** 2026-07-19
- **Subreddit:** r/LocalLLaMA
- **Author:** milpster
- **Quote:** "I assume the fixed chat template thing is no longer relevant for up to date ggufs?"

**What the tool does with this:** it cannot answer the question as asked — deciding "fixed" versus
"stale" means comparing against an upstream reference, which this tool deliberately does not ship.
What it does instead is make the file's own template inspectable offline: `--show-template` prints
the exact template bytes the file carries, and the checks report whether that template is
self-consistent. Two quants of the same model that differ in their embedded template are
distinguished by that output, which is what turns the question into something answerable by looking
rather than by trial-and-error `--chat-template-file` overrides.

Test: `test_two_quants_of_one_model_are_told_apart_by_their_templates` — builds two files that
differ only in the embedded template and asserts the tool reports a defect in one, not the other,
and prints each template verbatim.

---

## 3. A "fixed" template that emits a token the file's vocabulary does not contain

- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1voepnh/qwen_38_still_seem_to_have_that_random_stop/
- **Date:** 2026-08-14
- **Subreddit:** r/LocalLLaMA
- **Author:** T_rex2700
- **Quote:** "either issue with chat template (the \"fixed\" template just made the tool calling error worse)"

**What the tool does with this:** it extracts every special token the template emits and looks each
one up in this file's own `tokenizer.ggml.tokens`, reporting `token-not-in-vocab` (a `warning`)
with the offending token named — the concrete mechanism by which a replacement template makes tool
calling worse, since a token absent from the vocabulary is tokenised as ordinary text instead of a
control token.

Test: `test_tool_call_token_missing_from_this_files_vocabulary`.

---

## 4. One quant needs its header patched, another is already intact

- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1vlmh0b/deepseek_v4_flash_0731_at_27_ts_decode_on_strix/
- **Date:** 2026-08-11
- **Subreddit:** r/LocalLLaMA
- **Author:** stereohype
- **Quote:** "Skip header patching if you use the community Q2_K_S drafter (intact chat template). Only the older Q2K-Q8 file needs its header patched"

**What the tool does with this:** it reads the header of each file and reports, per file, whether
the embedded template is intact — here the older file's template never emits the model's EOS token
and never reads a message `role`, reported as `eos-not-emitted` and `no-role-dispatch`, while the
intact one reports neither. That is the "which of my downloaded files needs patching" question,
answered from the header without loading the weights.

Test: `test_older_quant_needs_patching_while_the_intact_one_does_not`.

---

## Severity and exit codes

Three of the four situations above surface as `warning`, not `error`, because the affected file
still loads and generates. Warnings do not fail the run by default: `no-generation-prompt` alone
fires on healthy, widely used models (the whole Mistral `[INST]` family), so making warnings exit 1
would report those models as broken. Pass `--strict` (equivalently `--fail-on warning`) to make
warnings fail too, for example in a CI step that gates a model release. See the README for the full
table.
