"""The CI workflow must fail loudly when the dev dependencies do not install.

A `pip install -e ".[dev]" 2>/dev/null || pip install -e .` fallback hides a
broken dev extra and carries on without jinja2, which silently turns the
jinja-agreement tests into skips -- a green CI run that proved less than it
claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


@pytest.fixture
def workflow_text() -> str:
    assert WORKFLOW.exists(), f"missing workflow {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def test_install_step_has_no_silencing_fallback(workflow_text: str) -> None:
    install_lines = [
        line
        for line in workflow_text.splitlines()
        if "pip install" in line and "[dev]" in line
    ]
    assert install_lines, "the workflow must install the dev extra"
    for line in install_lines:
        assert "||" not in line, f"dev install must not fall back: {line.strip()}"
        assert "2>/dev/null" not in line, f"dev install must not be silenced: {line!r}"


def test_dev_extra_provides_the_jinja_cross_check(workflow_text: str) -> None:
    """The dev extra is what makes the jinja agreement tests actually run."""
    pyproject = WORKFLOW.parents[2] / "pyproject.toml"
    assert "jinja2" in pyproject.read_text(encoding="utf-8")
    assert '[dev]' in workflow_text
