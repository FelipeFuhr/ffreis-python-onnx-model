from pathlib import Path

import httpx
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from application import create_application  # noqa: E402
from config import Settings  # noqa: E402

pytestmark = pytest.mark.integration


def _write_tiny_sum_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "tiny_sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


@pytest.mark.asyncio
async def test_real_model_pipeline_integration(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    _write_tiny_sum_model(model_path)

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")

    application = create_application(Settings())
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ping_response = await client.get("/ping")
        assert ping_response.status_code == 200

        invoke_response = await client.post(
            "/invocations",
            content=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert invoke_response.status_code == 200
        assert invoke_response.json() == [[6.0], [15.0]]
