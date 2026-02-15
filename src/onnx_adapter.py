"""ONNX Runtime adapter implementation."""

from __future__ import annotations

import importlib
import json
import os
from typing import Protocol

import numpy as np

from base_adapter import BaseAdapter
from config import Settings
from parsed_types import ParsedInput


class OnnxAdapter(BaseAdapter):
    """Inference adapter backed by ONNX Runtime."""

    def __init__(self: OnnxAdapter, settings: Settings) -> None:
        """Initialize and load an ONNX Runtime session.

        Parameters
        ----------
        settings : Settings
            Runtime settings used to discover and configure model loading.
        """
        self.settings = settings
        self.session: _OnnxSessionProtocol | None = None
        self.input_names: list[str] | None = None
        self.output_names: list[str] | None = None
        self._output_map: dict[str, str] | None = None
        self._load()

    def _load(self: OnnxAdapter) -> None:
        """Load ONNX model and runtime session from disk."""
        onnxruntime_module = importlib.import_module("onnxruntime")
        session_options = onnxruntime_module.SessionOptions()

        model_filename = self.settings.model_filename
        if model_filename:
            model_filename = os.path.basename(model_filename)
        if not model_filename:
            model_filename = "model.onnx"
        path = os.path.join(self.settings.model_dir, model_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")

        providers = [
            p.strip() for p in self.settings.onnx_providers.split(",") if p.strip()
        ]
        if self.settings.onnx_intra_op_threads > 0:
            session_options.intra_op_num_threads = self.settings.onnx_intra_op_threads
        if self.settings.onnx_inter_op_threads > 0:
            session_options.inter_op_num_threads = self.settings.onnx_inter_op_threads

        optimization_level = self.settings.onnx_graph_opt_level
        if optimization_level == "disable":
            session_options.graph_optimization_level = (
                onnxruntime_module.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        elif optimization_level == "basic":
            session_options.graph_optimization_level = (
                onnxruntime_module.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif optimization_level == "extended":
            session_options.graph_optimization_level = (
                onnxruntime_module.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        else:
            session_options.graph_optimization_level = (
                onnxruntime_module.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        self.session = onnxruntime_module.InferenceSession(
            path, sess_options=session_options, providers=providers
        )
        session = self.session
        self.input_names = [item.name for item in session.get_inputs()]
        self.output_names = [item.name for item in session.get_outputs()]

        if self.settings.onnx_output_map_json:
            raw_output_map = json.loads(self.settings.onnx_output_map_json)
            if not isinstance(raw_output_map, dict):
                raise ValueError("ONNX_OUTPUT_MAP_JSON must be a JSON object")
            self._output_map = {
                str(response_key): str(onnx_name)
                for response_key, onnx_name in raw_output_map.items()
            }

    def is_ready(self: OnnxAdapter) -> bool:
        """Return whether the runtime session and metadata are available."""
        return (
            self.session is not None
            and self.input_names is not None
            and self.output_names is not None
        )

    def _coerce(self: OnnxAdapter, arr: np.ndarray) -> np.ndarray:
        """Coerce arrays to common ONNX numeric dtypes.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array to cast when required.

        Returns
        -------
        numpy.ndarray
            Cast array.
        """
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32, copy=False)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64, copy=False)
        return arr

    def predict(self: OnnxAdapter, parsed_input: object) -> object:
        """Run inference for tabular or tensor input payloads.

        Parameters
        ----------
        parsed_input : object
            Parsed input containing either ``X`` features or named tensors.

        Returns
        -------
        object
            Prediction output represented as JSON-serializable structures.
        """
        if not isinstance(parsed_input, ParsedInput):
            raise TypeError("ONNX adapter expects ParsedInput")

        feed = self._build_feed(parsed_input)
        if self._output_map:
            return self._predict_with_output_map(feed)
        outputs = self._predict_single_output(feed)
        return outputs.tolist() if hasattr(outputs, "tolist") else outputs

    def _build_feed(
        self: OnnxAdapter, parsed_input: ParsedInput
    ) -> dict[str, np.ndarray]:
        """Build ONNX feed dictionary from parsed input."""
        if parsed_input.tensors is not None:
            return {
                key: self._coerce(np.asarray(value))
                for key, value in parsed_input.tensors.items()
            }
        if parsed_input.X is None:
            raise ValueError(
                "ONNX adapter requires ParsedInput.X or ParsedInput.tensors"
            )
        features = self._coerce(np.asarray(parsed_input.X))
        input_names = self.input_names
        if not input_names:
            raise RuntimeError("ONNX adapter input metadata is unavailable")
        return {self.settings.onnx_input_name or input_names[0]: features}

    def _predict_with_output_map(
        self: OnnxAdapter, feed: dict[str, np.ndarray]
    ) -> dict[str, object]:
        """Run inference and map outputs according to configured aliases."""
        output_map = self._output_map
        session = self.session
        if output_map is None or session is None:
            raise RuntimeError("ONNX output map or session not initialized")
        requested_outputs = list(output_map.values())
        outputs = session.run(requested_outputs, feed)
        mapped_outputs = {}
        for (response_key, _onnx_name), value in zip(
            output_map.items(), outputs, strict=False
        ):
            mapped_outputs[response_key] = (
                value.tolist() if hasattr(value, "tolist") else value
            )
        return mapped_outputs

    def _predict_single_output(
        self: OnnxAdapter, feed: dict[str, np.ndarray]
    ) -> object:
        """Run inference and return a single configured output tensor."""
        session = self.session
        output_names = self.output_names
        if session is None or output_names is None:
            raise RuntimeError("ONNX session not initialized")
        if self.settings.onnx_output_name:
            return session.run([self.settings.onnx_output_name], feed)[0]
        output_index = max(
            0,
            min(self.settings.onnx_output_index, len(output_names) - 1),
        )
        return session.run([output_names[output_index]], feed)[0]


class _OnnxInputOutputInfoProtocol(Protocol):
    """Protocol for ONNX input and output metadata."""

    name: str


class _OnnxSessionProtocol(Protocol):
    """Protocol for ONNX inference session."""

    def get_inputs(self: _OnnxSessionProtocol) -> list[_OnnxInputOutputInfoProtocol]:
        """Return model input metadata."""

    def get_outputs(self: _OnnxSessionProtocol) -> list[_OnnxInputOutputInfoProtocol]:
        """Return model output metadata."""

    def run(
        self: _OnnxSessionProtocol,
        output_names: list[str],
        feed: dict[str, np.ndarray],
    ) -> list[object]:
        """Run inference for output names and input feed."""
