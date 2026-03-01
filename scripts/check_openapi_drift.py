#!/usr/bin/env python3
"""Fail when HTTP API implementation changes without OpenAPI updates."""

from __future__ import annotations

from argparse import ArgumentParser as argparse_ArgumentParser
from pathlib import Path
from subprocess import CalledProcessError as subprocess_CalledProcessError
from subprocess import run as subprocess_run
from sys import stderr as sys_stderr

OPENAPI_FILE = "docs/openapi.yaml"
API_CONTRACT_SOURCES = ("src/application.py",)


def _git_changed_files(base: str, head: str) -> list[str]:
    result = subprocess_run(
        ["git", "diff", "--name-only", f"{base}..{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _matches_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix.rstrip('/')}/")


def main() -> int:
    """Validate that API code and OpenAPI file evolve together across two commits."""
    parser = argparse_ArgumentParser()
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    args = parser.parse_args()

    try:
        changed = _git_changed_files(args.base, args.head)
    except subprocess_CalledProcessError as exc:
        print(
            f"Failed to diff commits {args.base}..{args.head}: {exc}", file=sys_stderr
        )
        return 1

    api_changed = any(
        any(_matches_prefix(path, src) for src in API_CONTRACT_SOURCES)
        for path in changed
    )
    openapi_changed = any(_matches_prefix(path, OPENAPI_FILE) for path in changed)

    if api_changed and not openapi_changed:
        print("OpenAPI drift check failed.", file=sys_stderr)
        print(
            "Detected API implementation changes in src/application.py "
            "without updating docs/openapi.yaml.",
            file=sys_stderr,
        )
        print(
            "If the HTTP contract changed, update OpenAPI accordingly.", file=sys_stderr
        )
        print(
            "If it did not change, include a no-op/docs update "
            "or adjust the drift rules.",
            file=sys_stderr,
        )
        return 1

    if not Path(OPENAPI_FILE).exists():
        print(f"Required OpenAPI file not found: {OPENAPI_FILE}", file=sys_stderr)
        return 1

    print("OpenAPI drift check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
