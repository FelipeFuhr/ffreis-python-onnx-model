"""Tests for live server inference."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
uvicorn = pytest.importorskip("uvicorn")

from onnx import TensorProto, helper  # noqa: E402

from application import create_application  # noqa: E402
from config import Settings  # noqa: E402

pytestmark = pytest.mark.e2e


def _write_tiny_sum_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "tiny_sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _write_tiny_two_input_add_model(path: Path) -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["N", 1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, ["N", 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    add = helper.make_node("Add", inputs=["a", "b"], outputs=["y"])
    graph = helper.make_graph([add], "tiny_two_input_add_graph", [a, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.getenv(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def _run_live_server(base_url: str) -> Iterator[None]:
    application = create_application(Settings())
    host, port_text = base_url.replace("http://", "").split(":", 1)
    config = uvicorn.Config(
        app=application,
        host=host,
        port=int(port_text),
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    try:
        _wait_for_status(base_url, endpoint="/live", expected_status=200)
        yield
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def _wait_for_status(
    base_url: str, endpoint: str, expected_status: int, timeout_s: float = 15.0
) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = httpx.get(f"{base_url}{endpoint}", timeout=1.0)
            if response.status_code == expected_status:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server did not reach {endpoint}={expected_status} in time")


def test_live_server_end_to_end_inference(tmp_path: Path) -> None:
    """Verify live server end to end inference."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = {
        "SM_MODEL_DIR": str(tmp_path),
        "MODEL_TYPE": "onnx",
        "OTEL_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
        "CSV_HAS_HEADER": "false",
    }

    with _temporary_environment(environment):
        with _run_live_server(base_url):
            response = httpx.post(
                f"{base_url}/invocations",
                content=b"1,2,3\n4,5,6\n",
                headers={"Content-Type": "text/csv", "Accept": "application/json"},
                timeout=5.0,
            )
            assert response.status_code == 200
            assert response.json() == [[6.0], [15.0]]


def test_live_server_health_metrics_and_validation_errors(tmp_path: Path) -> None:
    """Verify health, metrics, and validation failures on live HTTP server."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = {
        "SM_MODEL_DIR": str(tmp_path),
        "MODEL_TYPE": "onnx",
        "OTEL_ENABLED": "false",
        "PROMETHEUS_ENABLED": "true",
        "CSV_HAS_HEADER": "false",
        "MAX_RECORDS": "1",
    }

    with _temporary_environment(environment):
        with _run_live_server(base_url):
            live_response = httpx.get(f"{base_url}/live", timeout=5.0)
            ready_response = httpx.get(f"{base_url}/ready", timeout=5.0)
            metrics_response = httpx.get(f"{base_url}/metrics", timeout=5.0)
            assert live_response.status_code == 200
            assert ready_response.status_code == 200
            assert metrics_response.status_code == 200
            assert (
                "byoc_up" in metrics_response.text or "# HELP" in metrics_response.text
            )

            too_many_records_response = httpx.post(
                f"{base_url}/invocations",
                content=b"1,2,3\n4,5,6\n",
                headers={"Content-Type": "text/csv", "Accept": "application/json"},
                timeout=5.0,
            )
            assert too_many_records_response.status_code == 400
            assert "too_many_records" in too_many_records_response.json()["error"]

            unsupported_content_type_response = httpx.post(
                f"{base_url}/invocations",
                content=b"<xml/>",
                headers={
                    "Content-Type": "application/xml",
                    "Accept": "application/json",
                },
                timeout=5.0,
            )
            assert unsupported_content_type_response.status_code == 400
            assert (
                "Unsupported Content-Type"
                in unsupported_content_type_response.json()["error"]
            )


def test_live_server_jsonl_multi_input_end_to_end(tmp_path: Path) -> None:
    """Verify JSONL multi-input request mapping against live HTTP server."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_two_input_add_model(model_path)

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = {
        "SM_MODEL_DIR": str(tmp_path),
        "MODEL_TYPE": "onnx",
        "OTEL_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
        "ONNX_INPUT_MAP_JSON": '{"feature_a":"a","feature_b":"b"}',
        "ONNX_INPUT_DTYPE_MAP_JSON": '{"a":"float32","b":"float32"}',
        "ONNX_DYNAMIC_BATCH": "true",
    }

    with _temporary_environment(environment):
        with _run_live_server(base_url):
            response = httpx.post(
                f"{base_url}/invocations",
                content=(
                    b'{"feature_a":[1.0],"feature_b":[2.0]}\n'
                    b'{"feature_a":[3.0],"feature_b":[4.0]}\n'
                ),
                headers={
                    "Content-Type": "application/x-ndjson",
                    "Accept": "application/json",
                },
                timeout=5.0,
            )
            assert response.status_code == 200
            assert response.json() == [[3.0], [7.0]]


def test_live_server_csv_header_auto_and_wrapped_json_output(tmp_path: Path) -> None:
    """Verify CSV header auto-detect and wrapped JSON output over live HTTP."""
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = {
        "SM_MODEL_DIR": str(tmp_path),
        "MODEL_TYPE": "onnx",
        "OTEL_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
        "CSV_HAS_HEADER": "auto",
        "RETURN_PREDICTIONS_ONLY": "false",
        "JSON_OUTPUT_KEY": "predictions",
    }

    with _temporary_environment(environment):
        with _run_live_server(base_url):
            response = httpx.post(
                f"{base_url}/invocations",
                content=b"f1,f2,f3\n1,2,3\n4,5,6\n",
                headers={"Content-Type": "text/csv", "Accept": "application/json"},
                timeout=5.0,
            )
            assert response.status_code == 200
            assert response.json() == {"predictions": [[6.0], [15.0]]}


def test_live_server_readiness_failure_with_missing_model(tmp_path: Path) -> None:
    """Verify readiness endpoint reports failure when model cannot be loaded."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = {
        "SM_MODEL_DIR": str(tmp_path),
        "MODEL_TYPE": "onnx",
        "OTEL_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
    }

    with _temporary_environment(environment):
        with _run_live_server(base_url):
            ready_response = httpx.get(f"{base_url}/ready", timeout=5.0)
            ping_response = httpx.get(f"{base_url}/ping", timeout=5.0)
            assert ready_response.status_code == 500
            assert ping_response.status_code == 500
