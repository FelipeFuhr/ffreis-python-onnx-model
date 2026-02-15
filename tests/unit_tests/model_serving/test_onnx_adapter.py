from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from config import Settings  # noqa: E402
from onnx_adapter import OnnxAdapter  # noqa: E402
from parsed_types import ParsedInput  # noqa: E402

pytestmark = pytest.mark.unit


def _write_sum_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0])
    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph([matmul], "sum_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _write_two_output_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, ["N", 2])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, ["N", 2])
    add = helper.make_node("Add", inputs=["x", "x"], outputs=["y1"])
    ident = helper.make_node("Identity", inputs=["x"], outputs=["y2"])
    graph = helper.make_graph([add, ident], "two_out_graph", [x], [y1, y2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


class TestOnnxAdapter:
    def test_raises_when_model_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            OnnxAdapter(Settings())

    def test_predicts_from_tabular_input(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.onnx"
        _write_sum_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(X=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        preds = adapter.predict(inp)
        assert preds == [[6.0], [15.0]]

    def test_predicts_from_tensor_inputs_with_output_map(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.onnx"
        _write_two_output_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        monkeypatch.setenv("ONNX_OUTPUT_MAP_JSON", '{"double":"y1","raw":"y2"}')
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(
            tensors={"x": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
        )
        preds = adapter.predict(inp)
        assert preds == {
            "double": [[2.0, 4.0], [6.0, 8.0]],
            "raw": [[1.0, 2.0], [3.0, 4.0]],
        }

    def test_selects_named_output(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.onnx"
        _write_two_output_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        monkeypatch.setenv("ONNX_OUTPUT_NAME", "y2")
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(X=np.asarray([[2.0, 4.0]], dtype=np.float32))
        preds = adapter.predict(inp)
        assert preds == [[2.0, 4.0]]

    def test_selects_output_by_index(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.onnx"
        _write_two_output_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        monkeypatch.setenv("ONNX_OUTPUT_INDEX", "1")
        adapter = OnnxAdapter(Settings())
        inp = ParsedInput(X=np.asarray([[2.0, 4.0]], dtype=np.float32))
        preds = adapter.predict(inp)
        assert preds == [[2.0, 4.0]]

    def test_rejects_empty_input(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.onnx"
        _write_sum_model(model_path)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        adapter = OnnxAdapter(Settings())
        with pytest.raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())
