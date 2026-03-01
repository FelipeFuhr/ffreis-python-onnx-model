#!/usr/bin/env python3
"""Validate OpenAPI contract and enforce runtime drift checks."""

from __future__ import annotations

from json import dumps as json_dumps
from pathlib import Path

from openapi_spec_validator import validate_spec
from yaml import safe_load as yaml_safe_load

from application import create_application
from config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_spec(path: Path) -> dict[str, object]:
    loaded = yaml_safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected OpenAPI mapping at {path}")
    return loaded


def _assert_unique_operation_ids(spec: dict[str, object]) -> None:
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        raise RuntimeError("OpenAPI 'paths' must be an object")
    seen: set[str] = set()
    for path_item in paths.values():
        if not isinstance(path_item, dict):
            continue
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if isinstance(operation_id, str):
                if operation_id in seen:
                    raise RuntimeError(f"Duplicate operationId: {operation_id}")
                seen.add(operation_id)


def _assert_runtime_matches_file(spec: dict[str, object]) -> None:
    generated = create_application(Settings()).openapi()
    if generated == spec:
        return
    expected = json_dumps(spec, indent=2, sort_keys=True)
    actual = json_dumps(generated, indent=2, sort_keys=True)
    raise RuntimeError(
        "Runtime OpenAPI schema differs from docs/openapi.yaml.\n"
        "Update routes/openapi settings or refresh docs/openapi.yaml.\n"
        f"Expected (docs):\n{expected[:3000]}\n\nActual (runtime):\n{actual[:3000]}"
    )


def main() -> None:
    """Run OpenAPI schema validation and runtime contract drift checks."""
    spec_path = REPO_ROOT / "docs" / "openapi.yaml"
    spec = _load_spec(spec_path)
    validate_spec(spec)
    _assert_unique_operation_ids(spec)
    _assert_runtime_matches_file(spec)
    print("OpenAPI contract is valid and matches runtime generation.")


if __name__ == "__main__":
    main()
