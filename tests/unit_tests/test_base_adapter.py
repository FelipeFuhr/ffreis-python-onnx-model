"""Test module."""

import os
from pathlib import Path
from typing import Self

import pytest

from base_adapter import BaseAdapter, load_adapter
from config import Settings

pytestmark = pytest.mark.unit


class _RaisesAdapter(BaseAdapter):
    def is_ready(self: Self) -> bool:
        """Run is ready.

        Returns
        -------
        bool
            Return value produced by helper logic in this test module.
        """
        return BaseAdapter.is_ready(self)

    def predict(self: Self, parsed_input: object) -> object:
        """Run predict.

        Parameters
        ----------
        parsed_input : object
            Parsed input payload object passed to adapter methods.

        Returns
        -------
        object
            Return value produced by helper logic in this test module.
        """
        return BaseAdapter.predict(self, parsed_input)


class TestBaseAdapter:
    """Test suite."""

    def test_base_methods_raise_not_implemented(self: Self) -> None:
        """Validate base methods raise not implemented.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        adapter = _RaisesAdapter()
        with pytest.raises(NotImplementedError):
            adapter.is_ready()
        with pytest.raises(NotImplementedError):
            adapter.predict(None)

    def test_load_adapter_uses_onnx_when_model_type_is_onnx(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate load adapter uses onnx when model type is onnx.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """

        class FakeOnnx:
            """Test suite."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import onnx_adapter as onnx_mod

        monkeypatch.setattr(onnx_mod, "OnnxAdapter", FakeOnnx)
        monkeypatch.setenv("MODEL_TYPE", "onnx")
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeOnnx)

    def test_load_adapter_uses_onnx_when_default_model_exists(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate load adapter uses onnx when default model exists.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : object
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"x")

        class FakeOnnx:
            """Test suite."""

            def __init__(self: Self, settings: object) -> None:
                self.settings = settings

        import onnx_adapter as onnx_mod

        monkeypatch.setattr(onnx_mod, "OnnxAdapter", FakeOnnx)
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        out = load_adapter(settings)
        assert isinstance(out, FakeOnnx)
        assert os.path.exists(model_path)

    def test_load_adapter_rejects_non_onnx_types(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate load adapter rejects non onnx types.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        settings = Settings()
        with pytest.raises(RuntimeError, match="not implemented"):
            load_adapter(settings)

    def test_load_adapter_requires_model_type_or_file(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validate load adapter requires model type or file.

        Parameters
        ----------
        monkeypatch : object
            Pytest monkeypatch fixture used to configure environment and runtime hooks.
        tmp_path : object
            Temporary directory path provided by pytest for filesystem test artifacts.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MODEL_TYPE", "")
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        with pytest.raises(RuntimeError, match="Set MODEL_TYPE=onnx"):
            load_adapter(settings)
