import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("uvicorn")

from onnx import TensorProto, helper  # noqa: E402

pytestmark = pytest.mark.e2e


def _write_tiny_sum_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "tiny_sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = cast(tuple[str, int], sock.getsockname())[1]
    sock.close()
    return port


def _wait_for_ping(base_url: str, timeout_s: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = httpx.get(f"{base_url}/ping", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("Server did not become ready in time")


def test_live_server_end_to_end_inference(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["SM_MODEL_DIR"] = str(tmp_path)
    env["MODEL_TYPE"] = "onnx"
    env["OTEL_ENABLED"] = "false"
    env["PROMETHEUS_ENABLED"] = "false"
    env["CSV_HAS_HEADER"] = "false"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "serving:application",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_ping(base_url)
        response = httpx.post(
            f"{base_url}/invocations",
            content=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
            timeout=5.0,
        )
        assert response.status_code == 200
        assert response.json() == [[6.0], [15.0]]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
