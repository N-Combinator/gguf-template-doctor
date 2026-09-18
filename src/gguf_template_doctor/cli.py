"""Command line interface for gguf-template-doctor.

Exit codes:
  0  the file was read and nothing at or above the failure threshold was found
  1  the file was read and the doctor reported a finding at that threshold
  2  the file could not be read: missing, not GGUF, truncated or malformed

The threshold is ERROR by default: a warning describes a template that is
likely wrong but still usable, and widely used healthy models do trigger one
(`no-generation-prompt` fires on the whole Mistral [INST] family), so warnings
must not make a CI step red unless the user asks for it with --strict.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from typing import Any, Sequence, TextIO

from . import __version__
from .doctor import Report, Severity, diagnose, summarize
from .errors import GgufError
from .reader import GgufType, parse_header

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_UNREADABLE = 2

_SEVERITY_LABEL = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.INFO: "info",
}

#: --fail-on thresholds, most severe first.  A finding makes the run fail when
#: its severity is at least as severe as the chosen threshold, so "error" (the
#: default) ignores warnings and info.
_SEVERITY_ORDER = (Severity.ERROR, Severity.WARNING, Severity.INFO)
_FAIL_ON_LEVELS = {
    _SEVERITY_LABEL[severity]: _SEVERITY_ORDER[: index + 1]
    for index, severity in enumerate(_SEVERITY_ORDER)
}
DEFAULT_FAIL_ON = "error"
#: What --strict means, in --fail-on terms.
STRICT_FAIL_ON = "warning"


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the output is valid JSON.

    GGUF FLOAT32/FLOAT64 metadata may legitimately hold NaN or +/-Infinity.
    ``json`` spells those as the bare literals ``NaN``/``Infinity``, which no
    strict JSON parser accepts, so render them as strings instead. The value
    stays visible in the report rather than being silently dropped.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


#: Arrays longer than this are reported by length instead of in full: a GGUF
#: vocabulary array is six figures long and would dwarf the report.
MAX_JSON_LIST_ITEMS = 64


def _summarize_long_list(value: Any) -> Any:
    """Replace an over-long list with a length-carrying summary object.

    Dropping the key entirely used to hide from the reader that the file has
    the key at all.  The summary keeps the fact, the length and a short
    preview, so `--list-metadata --json` never silently omits metadata.
    """
    if isinstance(value, list) and len(value) > MAX_JSON_LIST_ITEMS:
        return {
            "truncated": True,
            "length": len(value),
            "items": value[:MAX_JSON_LIST_ITEMS],
        }
    return value


def _write_text(out: TextIO, text: str) -> None:
    """Write *text* to *out*, degrading characters it cannot encode.

    A report carrying non-ASCII template text (or a non-ASCII metadata value)
    must not die with a UnicodeEncodeError because stdout was opened in a
    narrow encoding - a redirect to a file under LC_ALL=C is enough.  The
    unencodable characters are replaced; the report still prints.
    """
    try:
        out.write(text)
        return
    except UnicodeEncodeError:
        pass
    encoding = getattr(out, "encoding", None) or "ascii"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    out.write(safe)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gguf-template-doctor",
        description=(
            "Read a GGUF header offline, extract the embedded chat template and "
            "report template problems."
        ),
    )
    parser.add_argument("path", help="path to a .gguf file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON instead of text",
    )
    parser.add_argument(
        "--show-template",
        action="store_true",
        help="print the chat template source",
    )
    parser.add_argument(
        "--list-metadata",
        action="store_true",
        help="print every metadata key with its type and value",
    )
    parser.add_argument(
        "--fail-on",
        choices=sorted(_FAIL_ON_LEVELS),
        default=DEFAULT_FAIL_ON,
        metavar="{error,warning,info}",
        help=(
            "lowest severity that makes the run exit 1 (default: error, so "
            "warnings and info report but do not fail)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="same as --fail-on warning: let warnings exit 1 too",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def resolve_fail_on(fail_on: str, strict: bool) -> tuple[Severity, ...]:
    """Severities that make the run exit 1.

    ``--strict`` is a spelling of ``--fail-on warning``; when both are given the
    more demanding of the two wins, so ``--strict --fail-on info`` still fails
    on info rather than being quietly downgraded.
    """
    threshold = _SEVERITY_ORDER.index(_FAIL_ON_LEVELS[fail_on][-1])
    if strict:
        threshold = max(
            threshold, _SEVERITY_ORDER.index(_FAIL_ON_LEVELS[STRICT_FAIL_ON][-1])
        )
    return _SEVERITY_ORDER[: threshold + 1]


def _format_value(value: object) -> str:
    if isinstance(value, list):
        preview = ", ".join(repr(v) for v in value[:4])
        suffix = ", ..." if len(value) > 4 else ""
        return f"[{len(value)} items] {preview}{suffix}"
    if isinstance(value, str):
        if len(value) > 120 or "\n" in value:
            return f"<{len(value)} chars>"
        return repr(value)
    return repr(value)


def _print_metadata(gguf, out: TextIO) -> None:
    print("metadata:", file=out)
    for key, value in gguf.metadata.items():
        type_id = gguf.metadata_types.get(key)
        try:
            type_name = GgufType(type_id).name.lower()
        except ValueError:  # pragma: no cover - guarded by the parser
            type_name = str(type_id)
        print(
            f"  {key} ({type_name}) = {_format_value(value)}",
            file=out,
        )


def _print_text_report(
    report: Report, gguf, args: argparse.Namespace, out: TextIO
) -> None:
    print(f"file:         {report.path}", file=out)
    print(f"gguf version: {gguf.version}", file=out)
    print(f"architecture: {report.architecture or '(unset)'}", file=out)
    print(f"model:        {report.model_name or '(unset)'}", file=out)
    print(f"tensors:      {gguf.tensor_count}", file=out)
    print(f"metadata:     {gguf.kv_count} keys", file=out)

    for warning in report.parse_warnings:
        print(f"note: {warning}", file=out)

    if report.templates:
        names = ", ".join(t.name for t in report.templates)
        print(f"chat template(s): {names}", file=out)
        for template in report.templates:
            print(
                f"  {template.name}: {len(template.source)} chars "
                f"from {template.key}",
                file=out,
            )
    else:
        print("chat template(s): none", file=out)

    if args.list_metadata:
        _print_metadata(gguf, out)

    if args.show_template:
        for template in report.templates:
            print(f"--- template {template.name} ---", file=out)
            print(template.source, file=out)
            print(f"--- end template {template.name} ---", file=out)

    print("", file=out)
    if report.findings:
        print("findings:", file=out)
        for finding in report.findings:
            label = _SEVERITY_LABEL[finding.severity]
            scope = "" if finding.template == "-" else f"[{finding.template}] "
            print(
                f"  {label}: {scope}{finding.code}: {finding.message}",
                file=out,
            )
    else:
        print("findings: none", file=out)

    errors = len(report.errors)
    warnings = len(report.warnings)
    infos = len(report.findings) - errors - warnings
    print(
        f"summary: {errors} error(s), {warnings} warning(s), {infos} info",
        file=out,
    )
    # Say what the exit code is going to mean: a warning-only run exits 0 by
    # default, and a reader who sees "1 warning(s)" deserves to know why.
    failing = resolve_fail_on(args.fail_on, args.strict)
    if Severity.WARNING not in failing and warnings:
        print(
            f"note: exit code reflects errors only ({warnings} warning(s) "
            "ignored); use --strict to fail on warnings too",
            file=out,
        )


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    try:
        gguf = parse_header(args.path)
    except GgufError as exc:
        # Every anticipated failure lands here as a single readable line.
        _write_text(err, f"gguf-template-doctor: error: {exc}\n")
        return EXIT_UNREADABLE

    report = diagnose(gguf)
    failing = resolve_fail_on(args.fail_on, args.strict)
    exit_code = (
        EXIT_FINDINGS
        if any(f.severity in failing for f in report.findings)
        else EXIT_OK
    )

    buffer = io.StringIO()
    if args.json:
        payload = summarize(report)
        payload["gguf_version"] = gguf.version
        payload["tensor_count"] = gguf.tensor_count
        payload["metadata_key_count"] = gguf.kv_count
        # The severity policy is part of the answer: "ok" describes the file,
        # "exit_code" describes what this invocation decided about it.
        payload["fail_on"] = _SEVERITY_LABEL[failing[-1]]
        payload["exit_code"] = exit_code
        if args.show_template:
            payload["template_sources"] = {
                t.name: t.source for t in report.templates
            }
        if args.list_metadata:
            payload["metadata"] = {
                key: _summarize_long_list(value)
                for key, value in gguf.metadata.items()
            }
        # allow_nan=False makes any non-finite value we failed to convert a
        # loud ValueError instead of silently invalid JSON on stdout.
        json.dump(
            _json_safe(payload),
            buffer,
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        print("", file=buffer)
    else:
        _print_text_report(report, gguf, args, buffer)

    # One guarded write: stdout may be opened in an encoding that cannot hold
    # the template's characters, and that must not become a traceback.
    _write_text(out, buffer.getvalue())

    return exit_code


def entrypoint() -> None:  # pragma: no cover - thin console-script wrapper
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("gguf-template-doctor: interrupted", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
