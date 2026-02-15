import os

import pytest

from base_adapter import BaseAdapter, load_adapter
from config import Settings

pytestmark = pytest.mark.unit


class _RaisesAdapter(BaseAdapter):
    def is_ready(self) -> bool:
        return BaseAdapter.is_ready(self)

    def predict(self, parsed_input):
        return BaseAdapter.predict(self, parsed_input)


class TestBaseAdapter:
    def test_base_methods_raise_not_implemented(self):
        adapter = _RaisesAdapter()
        with pytest.raises(NotImplementedError):
            adapter.is_ready()
        with pytest.raises(NotImplementedError):
            adapter.predict(None)

    def test_load_adapter_uses_onnx_when_model_type_is_onnx(self, monkeypatch):
        class FakeOnnx:
            def __init__(self, settings):
                self.settings = settings

        import onnx_adapter as onnx_mod

        monkeypatch.setattr(onnx_mod, "OnnxAdapter", FakeOnnx)
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeOnnx)

    def test_load_adapter_uses_onnx_when_default_model_exists(
        self, monkeypatch, tmp_path
    ):
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"x")

        class FakeOnnx:
            def __init__(self, settings):
                self.settings = settings

        import onnx_adapter as onnx_mod

        monkeypatch.setattr(onnx_mod, "OnnxAdapter", FakeOnnx)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeOnnx)
        assert os.path.exists(model_path)

    def test_load_adapter_rejects_non_onnx_types(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        settings = Settings()
        with pytest.raises(RuntimeError, match="not implemented"):
            load_adapter(settings)

    def test_load_adapter_requires_model_type_or_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        with pytest.raises(RuntimeError, match="Set MODEL_TYPE=onnx"):
            load_adapter(settings)
