"""The single public command-line entry point for CDML."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdml",
        description="Detect commercial-break fades and write chapter markers.")
    parser.add_argument("command", nargs="?", help="infer, chapters, or model")
    parser.add_argument("subcommand", nargs="?", help="mark or download")
    return parser


def _run(handler: Callable[[Sequence[str] | None], None], args: Sequence[str]) -> None:
    handler(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the public ``cdml`` command without duplicating subcommand parsers."""
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        print("\nCommands:\n  infer [OPTIONS]\n  chapters mark INPUT [OPTIONS]\n  model download [OPTIONS]")
        return

    command, rest = args[0], args[1:]
    if command == "infer":
        from .infer import main as infer_main
        _run(infer_main, rest)
        return
    if command == "chapters":
        if not rest or rest[0] in {"-h", "--help"}:
            parser.error("usage: cdml chapters mark INPUT [OPTIONS]")
        if rest[0] == "mark":
            from .mark_chapters import main as mark_main
            _run(mark_main, rest[1:])
            return
    if command == "model":
        if not rest or rest[0] in {"-h", "--help"}:
            parser.error("usage: cdml model download [OPTIONS]")
        if rest[0] == "download":
            from .model_store import main as download_main
            _run(download_main, rest[1:])
            return
    parser.error(f"unknown command: {' '.join(args[:2])}")
