"""ONNX Runtime adapter implementation."""

from __future__ import annotations

import json
import os

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
        self.session = None
        self.input_names = None
        self.output_names = None
        self._output_map = None
        self._load()

    def _load(self: OnnxAdapter) -> None:
        """Load ONNX model and runtime session from disk."""
        import onnxruntime as ort

        model_filename = self.settings.model_filename or "model.onnx"
        path = os.path.join(self.settings.model_dir, model_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")

        providers = [
            p.strip() for p in self.settings.onnx_providers.split(",") if p.strip()
        ]
        session_options = ort.SessionOptions()

        if self.settings.onnx_intra_op_threads > 0:
            session_options.intra_op_num_threads = self.settings.onnx_intra_op_threads
        if self.settings.onnx_inter_op_threads > 0:
            session_options.inter_op_num_threads = self.settings.onnx_inter_op_threads

        optimization_level = self.settings.onnx_graph_opt_level
        if optimization_level == "disable":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        elif optimization_level == "basic":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif optimization_level == "extended":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        else:
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        self.session = ort.InferenceSession(
            path, sess_options=session_options, providers=providers
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        if self.settings.onnx_output_map_json:
            self._output_map = json.loads(self.settings.onnx_output_map_json)

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

    def predict(self: OnnxAdapter, parsed_input: ParsedInput) -> object:
        """Run inference for tabular or tensor input payloads.

        Parameters
        ----------
        parsed_input : ParsedInput
            Parsed input containing either ``X`` features or named tensors.

        Returns
        -------
        object
            Prediction output represented as JSON-serializable structures.
        """
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
        return {self.settings.onnx_input_name or self.input_names[0]: features}

    def _predict_with_output_map(
        self: OnnxAdapter, feed: dict[str, np.ndarray]
    ) -> dict[str, object]:
        """Run inference and map outputs according to configured aliases."""
        requested_outputs = list(self._output_map.values())
        outputs = self.session.run(requested_outputs, feed)
        mapped_outputs = {}
        for (response_key, _onnx_name), value in zip(
            self._output_map.items(), outputs, strict=False
        ):
            mapped_outputs[response_key] = (
                value.tolist() if hasattr(value, "tolist") else value
            )
        return mapped_outputs

    def _predict_single_output(
        self: OnnxAdapter, feed: dict[str, np.ndarray]
    ) -> object:
        """Run inference and return a single configured output tensor."""
        if self.settings.onnx_output_name:
            return self.session.run([self.settings.onnx_output_name], feed)[0]
        output_index = max(
            0,
            min(self.settings.onnx_output_index, len(self.output_names) - 1),
        )
        return self.session.run([self.output_names[output_index]], feed)[0]
