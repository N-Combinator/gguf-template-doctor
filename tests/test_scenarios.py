"""One test per public report that motivated this tool.

Each test builds a GGUF carrying exactly the template defect its report
describes, runs the CLI over it, and asserts the tool both finds the defect and
names the cause.  The reports themselves are recorded in docs/motivation.md;
each docstring quotes its source verbatim.
"""

from __future__ import annotations

import io

import pytest
from gguf_builder import ARRAY, STRING, UINT32, build_gguf

from gguf_template_doctor.cli import EXIT_FINDINGS, EXIT_OK, main


def run(*argv):
    """Invoke the CLI in-process, returning (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def write_model(path, template, *, arch, vocab, eos_id):
    path.write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, arch),
                ("tokenizer.ggml.model", STRING, "gpt2"),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, vocab)),
                ("tokenizer.ggml.eos_token_id", UINT32, eos_id),
                ("tokenizer.chat_template", STRING, template),
            ]
        )
    )
    return path


# --------------------------------------------------------------------------
# 1. andy2na, 2026-04-10
# --------------------------------------------------------------------------

GEMMA_VOCAB = ["<start_of_turn>", "<end_of_turn>", "<|tool_call|>", "hello"]

#: A tool-calling branch whose `{% if %}` is closed by the loop's `{% endfor %}`:
#: the shape of breakage a replacement .jinja file is shipped to repair.
GEMMA_BROKEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message.role + '\n' }}"
    "{% if message.tool_calls %}{{ '```tool_code\n' }}"
    "{{ message.content }}{{ '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)


def test_gemma_tool_calling_branch_is_structurally_broken(tmp_path):
    """andy2na, r/LocalLLaMA, 2026-04-10.

    "New chat templates from Google to fix tool calling" (linking a new .jinja
    file to replace the broken embedded one)

    https://www.reddit.com/r/LocalLLaMA/comments/1shs6sx/more_gemma4_fixes_in_the_past_24_hours/
    """
    path = write_model(
        tmp_path / "gemma-4-tool-calling.gguf",
        GEMMA_BROKEN_TEMPLATE,
        arch="gemma",
        vocab=GEMMA_VOCAB,
        eos_id=1,
    )

    code, out, err = run(str(path))

    assert code == EXIT_FINDINGS, out
    assert "error: [default] unbalanced-block" in out
    # The cause is named, not merely detected: which tag, closing what, where.
    assert "{% endfor %}" in out
    assert "closes a {% if %} block" in out
    assert "expected {% endif %}" in out
    assert "Traceback" not in err


def test_gemma_replacement_template_is_accepted(tmp_path):
    """The repaired form of the same template must come back clean.

    Guards the finding above against being an artefact of the fixture: the only
    difference here is the `{% endif %}` the broken template omits.
    """
    fixed = GEMMA_BROKEN_TEMPLATE.replace("{% endfor %}", "{% endif %}{% endfor %}")
    path = write_model(
        tmp_path / "gemma-4-fixed.gguf",
        fixed,
        arch="gemma",
        vocab=GEMMA_VOCAB,
        eos_id=1,
    )

    code, out, _ = run(str(path), "--strict")

    assert code == EXIT_OK, out
    assert "findings: none" in out


# --------------------------------------------------------------------------
# 2. milpster, 2026-07-19
# --------------------------------------------------------------------------

QWEN_VOCAB = ["<|im_start|>", "<|im_end|>", "hello", "world"]

#: The shape the fix has: the assistant turn header is appended only when the
#: server asks for it via add_generation_prompt.
QWEN_FIXED_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

#: The pre-fix shape: same loop, no add_generation_prompt branch at all.
QWEN_STALE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{% endfor %}"
)


def test_two_quants_of_one_model_are_told_apart_by_their_templates(tmp_path):
    """milpster, r/LocalLLaMA, 2026-07-19.

    "I assume the fixed chat template thing is no longer relevant for up to
    date ggufs?"

    https://www.reddit.com/r/LocalLLaMA/comments/1v0pp3s/qwen_36_27b_opencode_what_am_i_doing_wrong/

    The tool cannot say which template upstream considers "fixed" - that needs a
    reference table it deliberately does not ship (see docs/motivation.md).
    What it does is make the two files distinguishable offline: the stale one is
    reported with its defect named, the up-to-date one is clean, and each file's
    template is printed verbatim so the user can see which one they have.
    """
    stale = write_model(
        tmp_path / "qwen3.6-stale.gguf",
        QWEN_STALE_TEMPLATE,
        arch="qwen3",
        vocab=QWEN_VOCAB,
        eos_id=1,
    )
    fixed = write_model(
        tmp_path / "qwen3.6-uptodate.gguf",
        QWEN_FIXED_TEMPLATE,
        arch="qwen3",
        vocab=QWEN_VOCAB,
        eos_id=1,
    )

    stale_code, stale_out, _ = run(str(stale), "--strict")
    fixed_code, fixed_out, _ = run(str(fixed), "--strict")

    # The stale file is reported, and the reason is named.
    assert stale_code == EXIT_FINDINGS, stale_out
    assert "warning: [default] no-generation-prompt" in stale_out
    assert "add_generation_prompt" in stale_out
    # The up-to-date file is not.
    assert fixed_code == EXIT_OK, fixed_out
    assert "no-generation-prompt" not in fixed_out

    # And the templates themselves are inspectable, which is what lets a user
    # answer "which of my downloads is which" by reading rather than guessing.
    assert QWEN_STALE_TEMPLATE in run(str(stale), "--show-template")[1]
    assert QWEN_FIXED_TEMPLATE in run(str(fixed), "--show-template")[1]


# --------------------------------------------------------------------------
# 3. T_rex2700, 2026-08-14
# --------------------------------------------------------------------------

#: A replacement template that adds a tool-calling branch emitting a control
#: token this file's own vocabulary never defines.
QWEN_TOOL_CALL_TEMPLATE = (
    QWEN_FIXED_TEMPLATE
    + "{% if tools %}{{ '<|tool_call_start|>' }}{% endif %}"
)


def test_tool_call_token_missing_from_this_files_vocabulary(tmp_path):
    """T_rex2700, r/LocalLLaMA, 2026-08-14.

    "either issue with chat template (the \\"fixed\\" template just made the tool
    calling error worse)"

    https://www.reddit.com/r/LocalLLaMA/comments/1voepnh/qwen_38_still_seem_to_have_that_random_stop/
    """
    path = write_model(
        tmp_path / "qwen3.8-tool-call.gguf",
        QWEN_TOOL_CALL_TEMPLATE,
        arch="qwen3",
        vocab=QWEN_VOCAB,
        eos_id=1,
    )

    code, out, err = run(str(path), "--strict")

    assert code == EXIT_FINDINGS, out
    assert "warning: [default] token-not-in-vocab" in out
    # The cause is named: which token, and against which metadata key.
    assert "<|tool_call_start|>" in out
    assert "tokenizer.ggml.tokens" in out
    assert "Traceback" not in err
    # A warning on its own leaves the default run green; only --strict fails.
    assert run(str(path))[0] == EXIT_OK


# --------------------------------------------------------------------------
# 4. stereohype, 2026-08-11
# --------------------------------------------------------------------------

DEEPSEEK_VOCAB = [
    "<|User|>",
    "<|Assistant|>",
    "<|end▁of▁sentence|>",
    "hello",
]

#: The older file's header: one flat loop that never dispatches on role and
#: never emits the model's end-of-sentence token.
DEEPSEEK_UNPATCHED_TEMPLATE = (
    "{% for message in messages %}{{ '<|User|>' + message.content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|Assistant|>' }}{% endif %}"
)

#: The intact one: roles dispatched, end-of-sentence emitted after each turn.
DEEPSEEK_INTACT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'user' %}{{ '<|User|>' + message.content }}"
    "{% else %}"
    "{{ '<|Assistant|>' + message.content + '<|end▁of▁sentence|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|Assistant|>' }}{% endif %}"
)


def test_older_quant_needs_patching_while_the_intact_one_does_not(tmp_path):
    """stereohype, r/LocalLLaMA, 2026-08-11.

    "Skip header patching if you use the community Q2_K_S drafter (intact chat
    template). Only the older Q2K-Q8 file needs its header patched"

    https://www.reddit.com/r/LocalLLaMA/comments/1vlmh0b/deepseek_v4_flash_0731_at_27_ts_decode_on_strix/
    """
    older = write_model(
        tmp_path / "deepseek-v4-q2k-q8.gguf",
        DEEPSEEK_UNPATCHED_TEMPLATE,
        arch="deepseek2",
        vocab=DEEPSEEK_VOCAB,
        eos_id=2,
    )
    intact = write_model(
        tmp_path / "deepseek-v4-q2-k-s.gguf",
        DEEPSEEK_INTACT_TEMPLATE,
        arch="deepseek2",
        vocab=DEEPSEEK_VOCAB,
        eos_id=2,
    )

    older_code, older_out, _ = run(str(older), "--strict")
    intact_code, intact_out, _ = run(str(intact), "--strict")

    assert older_code == EXIT_FINDINGS, older_out
    # Both reasons the older header needs patching are named.
    assert "warning: [default] eos-not-emitted" in older_out
    assert "<|end▁of▁sentence|>" in older_out
    assert "warning: [default] no-role-dispatch" in older_out
    assert "`role`" in older_out

    assert intact_code == EXIT_OK, intact_out
    assert "findings: none" in intact_out


@pytest.mark.parametrize(
    "template",
    [
        GEMMA_BROKEN_TEMPLATE,
        QWEN_FIXED_TEMPLATE,
        QWEN_TOOL_CALL_TEMPLATE,
        DEEPSEEK_UNPATCHED_TEMPLATE,
        DEEPSEEK_INTACT_TEMPLATE,
    ],
)
def test_scenario_templates_are_what_jinja_says_they_are(template):
    """The scenario fixtures must be real Jinja, broken only where intended.

    Without this, a typo in a fixture could make a test pass for the wrong
    reason - a template Jinja itself rejects would trip the structural checks
    regardless of the defect the scenario is about.
    """
    jinja2 = pytest.importorskip("jinja2")

    expect_valid = template is not GEMMA_BROKEN_TEMPLATE
    try:
        jinja2.Environment().parse(template)
        parsed = True
    except jinja2.TemplateSyntaxError:
        parsed = False
    assert parsed is expect_valid
