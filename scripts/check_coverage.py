#!/usr/bin/env python3
"""Check Cobertura/XML line coverage against a minimum with optional downward drift."""

from __future__ import annotations

from os import environ as os_environ
from pathlib import Path
from sys import exit as sys_exit
from xml.etree.ElementTree import parse as ET_parse


def _parse_float(name: str, default: float) -> float:
    raw = os_environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid float for {name}: {raw!r}") from exc


def main() -> int:
    """Validate XML line-rate against configured minimum and drift floor."""
    label = os_environ.get("COVERAGE_LABEL", "coverage")
    xml_path = Path(os_environ.get("COVERAGE_XML", "coverage.xml"))
    minimum = _parse_float("COVERAGE_MIN", 0.0)
    drift = _parse_float("COVERAGE_MAX_DRIFT", 0.0)

    if drift < 0:
        raise SystemExit("COVERAGE_MAX_DRIFT must be >= 0")

    if not xml_path.exists():
        raise SystemExit(f"Coverage XML not found: {xml_path}")

    root = ET_parse(xml_path).getroot()
    raw_line_rate = root.attrib.get("line-rate")
    if raw_line_rate is None:
        raise SystemExit(f"line-rate attribute missing in {xml_path}")

    try:
        got = float(raw_line_rate) * 100.0
    except ValueError as exc:
        raise SystemExit(
            f"Invalid line-rate value in {xml_path}: {raw_line_rate!r}"
        ) from exc

    floor = max(minimum - drift, 0.0)
    print(
        f"{label}: got {got:.2f}% | target >= {minimum:.2f}% (drift floor {floor:.2f}%)"
    )

    if got + 1e-9 < floor:
        return 1
    return 0


if __name__ == "__main__":
    sys_exit(main())
