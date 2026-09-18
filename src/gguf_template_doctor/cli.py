"""Command line interface for gguf-template-doctor.

Exit codes:
  0  the file was read and no problems worse than INFO were found
  1  the file was read but the doctor reported warnings or errors
  2  the file could not be read: missing, not GGUF, truncated or malformed
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

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
        "--strict",
        action="store_true",
        help="exit 1 on INFO findings as well as warnings and errors",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _format_value(value: object, type_id: int | None) -> str:
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
            f"  {key} ({type_name}) = {_format_value(value, type_id)}",
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
        print(f"gguf-template-doctor: error: {exc}", file=err)
        return EXIT_UNREADABLE

    report = diagnose(gguf)

    if args.json:
        payload = summarize(report)
        payload["gguf_version"] = gguf.version
        payload["tensor_count"] = gguf.tensor_count
        payload["metadata_key_count"] = gguf.kv_count
        if args.show_template:
            payload["template_sources"] = {
                t.name: t.source for t in report.templates
            }
        if args.list_metadata:
            payload["metadata"] = {
                key: value
                for key, value in gguf.metadata.items()
                # Long arrays would dwarf the report; report their length instead.
                if not (isinstance(value, list) and len(value) > 64)
            }
        json.dump(payload, out, indent=2, sort_keys=True, default=str)
        print("", file=out)
    else:
        _print_text_report(report, gguf, args, out)

    if report.errors or report.warnings:
        return EXIT_FINDINGS
    if args.strict and report.findings:
        return EXIT_FINDINGS
    return EXIT_OK


def entrypoint() -> None:  # pragma: no cover - thin console-script wrapper
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("gguf-template-doctor: interrupted", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
