"""Shared utilities for training and serving ONNX example models."""

from __future__ import annotations

import importlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import httpx
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class ModelProtocol(Protocol):
    """Protocol for trainable sklearn-like models."""

    def fit(self: ModelProtocol, features: np.ndarray, labels: object) -> object:
        """Fit model."""

    def predict(self: ModelProtocol, features: np.ndarray) -> np.ndarray:
        """Predict labels."""


@dataclass(frozen=True)
class ExampleResult:
    """Summarized result for one train-and-serve example run.

    Parameters
    ----------
    algorithm_name : str
        Algorithm used for training.
    model_path : pathlib.Path
        Exported ONNX model path.
    request_row_count : int
        Number of rows sent to the inference endpoint.
    max_abs_difference : float
        Maximum absolute difference between sklearn and served predictions.
    """

    algorithm_name: str
    model_path: Path
    request_row_count: int
    max_abs_difference: float


def train_model(algorithm_name: str) -> tuple[ModelProtocol, np.ndarray]:
    """Train a sklearn model on Iris and return model plus features.

    Parameters
    ----------
    algorithm_name : str
        Model algorithm name. One of ``logistic_regression``, ``random_forest``,
        or ``neural_network``.

    Returns
    -------
    tuple[ModelProtocol, numpy.ndarray]
        Trained model and full training feature matrix.
    """
    datasets_module = importlib.import_module("sklearn.datasets")
    linear_model_module = importlib.import_module("sklearn.linear_model")
    ensemble_module = importlib.import_module("sklearn.ensemble")
    neural_network_module = importlib.import_module("sklearn.neural_network")

    iris = datasets_module.load_iris()
    features = iris.data.astype(np.float32)
    labels = iris.target

    if algorithm_name == "logistic_regression":
        model = linear_model_module.LogisticRegression(max_iter=300)
    elif algorithm_name == "random_forest":
        model = ensemble_module.RandomForestClassifier(
            n_estimators=200, random_state=42
        )
    elif algorithm_name == "neural_network":
        model = neural_network_module.MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=600,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    model.fit(features, labels)
    return model, features


def export_model_to_onnx(
    model: ModelProtocol, feature_count: int, output_path: Path
) -> Path:
    """Export a sklearn model to ONNX.

    Parameters
    ----------
    model : ModelProtocol
        Trained sklearn model.
    feature_count : int
        Number of input features.
    output_path : pathlib.Path
        Destination ONNX file path.

    Returns
    -------
    pathlib.Path
        Written ONNX model path.
    """
    skl2onnx_module = importlib.import_module("skl2onnx")
    data_types_module = importlib.import_module("skl2onnx.common.data_types")
    convert_sklearn = skl2onnx_module.convert_sklearn
    float_tensor_type = data_types_module.FloatTensorType

    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("x", float_tensor_type([None, feature_count]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.write_bytes(onnx_model.SerializeToString())
    return output_path


@contextmanager
def temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables.

    Parameters
    ----------
    overrides : dict[str, str]
        Environment variables to set while context is active.
    """
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


def _find_free_port() -> int:
    """Allocate an available localhost TCP port.

    Returns
    -------
    int
        Free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def run_serving_process(model_directory: Path) -> Iterator[str]:
    """Start serving API process and yield base URL.

    Parameters
    ----------
    model_directory : pathlib.Path
        Directory containing ``model.onnx``.

    Yields
    ------
    str
        Base URL for the running local serving API.
    """
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    environment = os.environ.copy()
    environment.update(
        {
            "SM_MODEL_DIR": str(model_directory),
            "MODEL_TYPE": "onnx",
            "OTEL_ENABLED": "false",
            "PROMETHEUS_ENABLED": "false",
            "CSV_HAS_HEADER": "false",
            "DEFAULT_ACCEPT": "application/json",
            "PYTHONPATH": str(SOURCE_ROOT),
        }
    )

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
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_until_server_ready(process, base_url)
        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _wait_until_server_ready(process: subprocess.Popen[str], base_url: str) -> None:
    """Wait until serving API responds to liveness check.

    Parameters
    ----------
    process : subprocess.Popen[str]
        Running serving process.
    base_url : str
        Local API base URL.
    """
    with httpx.Client(timeout=1.0) as client:
        for _ in range(60):
            exit_code = process.poll()
            if exit_code is not None:
                logs = process.stdout.read() if process.stdout is not None else ""
                raise SystemExit(
                    "FAIL: serving process exited before readiness "
                    f"(exit_code={exit_code}). Logs:\n{logs}"
                )
            try:
                response = client.get(f"{base_url}/live")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    raise SystemExit("FAIL: serving process did not become live in time.")


def _to_label_vector(raw_predictions: list[object], expected_rows: int) -> np.ndarray:
    """Normalize served predictions to a 1D numeric label vector.

    Parameters
    ----------
    raw_predictions : list[object]
        JSON-decoded prediction payload.
    expected_rows : int
        Expected number of predictions.

    Returns
    -------
    numpy.ndarray
        One-dimensional label vector.
    """
    prediction_array = np.asarray(raw_predictions)
    if prediction_array.size == 0:
        raise SystemExit("FAIL: invocation returned no predictions.")
    if prediction_array.ndim == 2 and prediction_array.shape[1] == 1:
        prediction_array = prediction_array[:, 0]
    if prediction_array.ndim != 1:
        raise SystemExit(
            "FAIL: expected vector-like predictions, "
            f"got shape={prediction_array.shape}."
        )
    if prediction_array.shape[0] != expected_rows:
        raise SystemExit(
            "FAIL: prediction count mismatch "
            f"(expected={expected_rows}, got={prediction_array.shape[0]})."
        )
    if not np.issubdtype(prediction_array.dtype, np.number):
        raise SystemExit(
            f"FAIL: expected numeric predictions, got dtype={prediction_array.dtype}."
        )
    return prediction_array.astype(np.float64)


def run_train_and_serve_demo(algorithm_name: str) -> ExampleResult:
    """Train, export, and serve an ONNX model for one algorithm.

    Parameters
    ----------
    algorithm_name : str
        Model algorithm name.

    Returns
    -------
    ExampleResult
        Example execution summary.
    """
    model, features = train_model(algorithm_name)
    model_directory = REPOSITORY_ROOT / "tmp" / "examples" / algorithm_name
    model_path = export_model_to_onnx(
        model=model,
        feature_count=features.shape[1],
        output_path=model_directory / "model.onnx",
    )
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise SystemExit(f"FAIL: ONNX model was not created at {model_path}.")

    batch = features[:8]
    sklearn_predictions = np.asarray(model.predict(batch), dtype=np.float64).reshape(-1)

    live_code, ready_code, invocation_code, raw_predictions = run_inference_request(
        model_directory, batch
    )
    served_predictions = _to_label_vector(raw_predictions, expected_rows=batch.shape[0])

    if live_code != 200 or ready_code != 200 or invocation_code != 200:
        raise SystemExit(
            "FAIL: health/invocation status codes were unexpected "
            f"(live={live_code}, ready={ready_code}, invocations={invocation_code})."
        )

    if sklearn_predictions.shape != served_predictions.shape:
        raise SystemExit(
            "FAIL: sklearn/served shape mismatch "
            f"(sklearn={sklearn_predictions.shape}, "
            f"served={served_predictions.shape})."
        )

    max_abs_difference = float(np.max(np.abs(sklearn_predictions - served_predictions)))
    if not np.array_equal(sklearn_predictions, served_predictions):
        raise SystemExit(
            "FAIL: prediction mismatch between sklearn and serving "
            f"(max_abs_diff={max_abs_difference:.6f})."
        )

    print(
        "PASS: "
        f"algorithm={algorithm_name} "
        f"model={model_path} "
        f"rows={batch.shape[0]} "
        f"max_abs_diff={max_abs_difference:.6f}"
    )
    return ExampleResult(
        algorithm_name=algorithm_name,
        model_path=model_path,
        request_row_count=int(batch.shape[0]),
        max_abs_difference=max_abs_difference,
    )


def run_inference_request(
    model_directory: Path, rows: np.ndarray
) -> tuple[int, int, int, list[object]]:
    """Run an inference request against the local serving process.

    Parameters
    ----------
    model_directory : pathlib.Path
        Directory containing ``model.onnx``.
    rows : numpy.ndarray
        Feature rows used to build CSV invocation payload.

    Returns
    -------
    tuple[int, int, int, list[object]]
        ``(live_status, ready_status, invocations_status, predictions_json)``.
    """
    payload_text = "\n".join(",".join(map(str, row.tolist())) for row in rows)
    payload = payload_text.encode("utf-8")
    with run_serving_process(model_directory) as base_url:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            liveness_response = client.get("/live")
            readiness_response = client.get("/ready")
            invocation_response = client.post(
                "/invocations",
                content=payload,
                headers={"Content-Type": "text/csv", "Accept": "application/json"},
            )
            if invocation_response.status_code != 200:
                raise SystemExit(
                    "FAIL: invocation failed "
                    f"(status={invocation_response.status_code}, "
                    f"body={invocation_response.text})."
                )
            return (
                liveness_response.status_code,
                readiness_response.status_code,
                invocation_response.status_code,
                cast(list[object], invocation_response.json()),
            )
