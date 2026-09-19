"""The release workflow must not be able to ship a mislabelled artifact.

A tag is the only thing a user sees before downloading; if ``v1.2.0`` builds a
wheel whose metadata says ``0.1.0``, nothing downstream notices.  The guard is
``scripts/check_version_tag.py``, and the substantive tests here run that script
directly against real trees rather than trusting the workflow yaml to be right.

The yaml-shape tests need PyYAML, which arrives only as a transitive dependency
of the ``gguf`` dev extra; they skip when it is absent, in the same way
``test_jinja_agreement.py`` skips without jinja2.  The script tests never skip.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CHECKER = ROOT / "scripts" / "check_version_tag.py"

sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def workflow_text() -> str:
    assert WORKFLOW.exists(), f"missing workflow {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture
def workflow() -> dict:
    """The release workflow, parsed.

    ``on:`` comes back as the boolean ``True``: YAML 1.1 reads the bare word as a
    truth value, and PyYAML still does.  The key is looked up as ``True`` rather
    than renaming the trigger, which GitHub requires to be spelled ``on``.
    """
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(workflow: dict) -> list[dict]:
    jobs = workflow["jobs"]
    assert len(jobs) == 1, f"expected a single release job, got {sorted(jobs)}"
    return list(jobs.values())[0]["steps"]


def _run_checker(tag: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), tag],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The version guard, exercised for real.
# --------------------------------------------------------------------------


def _fake_tree(tmp_path: Path, pyproject_version: str, dunder_version: str) -> Path:
    """Build the two files the checker reads, with the versions asked for."""
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\n'
        'requires = ["setuptools>=68"]\n'
        "\n"
        "[project]\n"
        'name = "gguf-template-doctor"\n'
        f'version = "{pyproject_version}"\n',
        encoding="utf-8",
    )
    package = tmp_path / "src" / "gguf_template_doctor"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f'__version__ = "{dunder_version}"\n', encoding="utf-8"
    )
    return tmp_path


def test_matching_tag_is_accepted(tmp_path: Path) -> None:
    tree = _fake_tree(tmp_path, "1.2.3", "1.2.3")
    import check_version_tag

    assert check_version_tag.check("v1.2.3", root=tree) == []


def test_tag_ahead_of_pyproject_is_rejected(tmp_path: Path) -> None:
    """The case the issue names: the tag says one thing, pyproject another."""
    tree = _fake_tree(tmp_path, "0.1.0", "0.1.0")
    import check_version_tag

    problems = check_version_tag.check("v1.2.3", root=tree)
    assert problems, "a tag ahead of pyproject.toml must be reported"
    assert any("pyproject.toml" in p for p in problems)
    assert any("'0.1.0'" in p and "'1.2.3'" in p for p in problems)


def test_package_dunder_version_is_checked_too(tmp_path: Path) -> None:
    """--version must not disagree with the wheel the tag built."""
    tree = _fake_tree(tmp_path, "1.2.3", "1.2.2")
    import check_version_tag

    problems = check_version_tag.check("v1.2.3", root=tree)
    assert len(problems) == 1, problems
    assert "__version__" in problems[0]


@pytest.mark.parametrize(
    "tag", ["1.2.3", "v1.2", "v1.2.3.4", "release-1.2.3", "v1.2.3-rc1", "v"]
)
def test_malformed_tags_are_rejected(tmp_path: Path, tag: str) -> None:
    tree = _fake_tree(tmp_path, "1.2.3", "1.2.3")
    import check_version_tag

    problems = check_version_tag.check(tag, root=tree)
    assert problems, f"{tag!r} is not a vX.Y.Z release tag"
    assert "vX.Y.Z" in problems[0]


def _tree_with_checker(tmp_path: Path) -> Path:
    """Copy the checker into *tmp_path* so it resolves that tree as its root."""
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    copy = scripts / "check_version_tag.py"
    copy.write_text(CHECKER.read_text(encoding="utf-8"), encoding="utf-8")
    return copy


def test_unreadable_version_is_reported_not_raised(tmp_path: Path) -> None:
    """A pyproject with no version is a named failure, not a traceback."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    package = tmp_path / "src" / "gguf_template_doctor"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    import check_version_tag

    with pytest.raises(check_version_tag.VersionReadError):
        check_version_tag.check("v1.2.3", root=tmp_path)

    copy = _tree_with_checker(tmp_path)
    result = subprocess.run(
        [sys.executable, str(copy), "v1.2.3"], capture_output=True, text=True
    )
    assert result.returncode == 2, result.stderr
    assert "cannot read the declared version" in result.stderr
    assert "Traceback" not in result.stderr


def test_non_utf8_pyproject_is_reported_not_raised(tmp_path: Path) -> None:
    """A latin-1 byte in pyproject.toml must not surface as a UnicodeDecodeError."""
    (tmp_path / "pyproject.toml").write_bytes(
        b'[project]\nname = "caf\xe9"\nversion = "1.2.3"\n'
    )
    package = tmp_path / "src" / "gguf_template_doctor"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")

    import check_version_tag

    with pytest.raises(check_version_tag.VersionReadError, match="not valid UTF-8"):
        check_version_tag.check("v1.2.3", root=tmp_path)


def test_checker_accepts_a_pyproject_with_a_bom(tmp_path: Path) -> None:
    """A BOM in front of `[build-system]` must not hide the version."""
    (tmp_path / "pyproject.toml").write_bytes(
        b"\xef\xbb\xbf" + b'[project]\nversion = "1.2.3"\n'
    )
    package = tmp_path / "src" / "gguf_template_doctor"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")

    import check_version_tag

    assert check_version_tag.check("v1.2.3", root=tmp_path) == []


def test_checker_accepts_the_real_repository_tag(tmp_path: Path) -> None:
    """The script run in place against its own tree agrees with its version."""
    import check_version_tag

    version = check_version_tag.read_pyproject_version(ROOT)
    result = _run_checker(f"v{version}", cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "matches the packaged version" in result.stdout


def test_checker_agrees_with_this_repository() -> None:
    """The versions committed right now are consistent with each other."""
    import check_version_tag

    assert check_version_tag.read_pyproject_version(ROOT) == (
        check_version_tag.read_package_version(ROOT)
    )


def test_checker_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    """The workflow relies on the exit code, so pin it end to end."""
    tree = _fake_tree(tmp_path, "0.1.0", "0.1.0")
    script = tree / "scripts"
    script.mkdir()
    (script / "check_version_tag.py").write_text(
        CHECKER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(script / "check_version_tag.py"), "v1.2.3"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "does not match" in result.stderr


# --------------------------------------------------------------------------
# The workflow wiring.
# --------------------------------------------------------------------------


def test_triggers_only_on_version_tags(workflow: dict) -> None:
    triggers = workflow[True]
    assert set(triggers) == {"push"}, f"release must only run on a tag push: {triggers}"
    tags = triggers["push"]["tags"]
    assert tags == ["v[0-9]+.[0-9]+.[0-9]+"], tags
    assert "branches" not in triggers["push"]


def test_runs_the_tests_before_building(workflow: dict) -> None:
    steps = _steps(workflow)
    runs = [step.get("run", "") for step in steps]
    test_at = next(i for i, run in enumerate(runs) if "pytest" in run)
    build_at = next(i for i, run in enumerate(runs) if "python -m build" in run)
    assert test_at < build_at, "the suite must run before the artifacts are built"


def test_tests_run_verbosely(workflow: dict) -> None:
    """`pytest -q` hides how many tests actually ran."""
    runs = [step.get("run", "") for step in _steps(workflow)]
    pytest_runs = [run for run in runs if "pytest" in run]
    assert pytest_runs, "the release workflow must run the tests"
    for run in pytest_runs:
        assert "-v" in run.split(), run
        assert "-q" not in run.split(), run


def test_version_check_runs_before_the_build(workflow: dict) -> None:
    steps = _steps(workflow)
    runs = [step.get("run", "") for step in steps]
    check_at = next(i for i, run in enumerate(runs) if "check_version_tag.py" in run)
    build_at = next(i for i, run in enumerate(runs) if "python -m build" in run)
    assert check_at < build_at, "a mismatched tag must stop before anything is built"


def test_version_check_uses_the_pushed_tag(workflow_text: str) -> None:
    assert "GITHUB_REF_NAME" in workflow_text


def test_release_attaches_both_artifacts(workflow: dict) -> None:
    """Criterion 2: wheel *and* sdist end up on the Release."""
    steps = _steps(workflow)
    release = next(
        step for step in steps if step.get("uses", "").startswith("softprops/action-gh-release")
    )
    files = release["with"]["files"]
    patterns = [line.strip() for line in files.strip().splitlines()]
    assert "dist/*.whl" in patterns, patterns
    assert "dist/*.tar.gz" in patterns, patterns
    assert release["with"]["fail_on_unmatched_files"] is True


def test_release_job_can_write_releases(workflow: dict) -> None:
    job = list(workflow["jobs"].values())[0]
    assert job["permissions"] == {"contents": "write"}


def test_actions_are_pinned_to_a_major_version(workflow: dict) -> None:
    for step in _steps(workflow):
        uses = step.get("uses")
        if uses is not None:
            assert "@" in uses, f"unpinned action: {uses}"


def test_ci_workflow_also_runs_pytest_verbosely() -> None:
    """Criterion 4, on the workflow the issue actually names."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    pytest_lines = [line for line in text.splitlines() if "pytest" in line and "run:" in line]
    assert pytest_lines, "ci.yml must run pytest"
    for line in pytest_lines:
        assert "-v" in line.split(), line
        assert "-q" not in line.split(), line


def test_pytest_addopts_does_not_force_quiet() -> None:
    """`-q` in addopts fights the `-v` the workflows pass."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'addopts = "-q"' not in pyproject
