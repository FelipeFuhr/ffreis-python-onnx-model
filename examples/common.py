"""Shared utilities for training and serving ONNX example models."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast

import httpx
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application import create_application  # noqa: E402
from config import Settings  # noqa: E402


class ModelProtocol(Protocol):
    """Protocol for trainable sklearn-like models."""

    def fit(self: ModelProtocol, features: np.ndarray, labels: object) -> object:
        """Fit model."""


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


async def run_inference_request(
    model_directory: Path, rows: np.ndarray
) -> list[object]:
    """Run an inference request through the real HTTP application.

    Parameters
    ----------
    model_directory : pathlib.Path
        Directory containing ``model.onnx``.
    rows : numpy.ndarray
        Feature rows used to build CSV invocation payload.

    Returns
    -------
    list[object]
        Parsed JSON predictions.
    """
    payload = "\n".join(",".join(map(str, row.tolist())) for row in rows).encode(
        "utf-8"
    )
    environment = {
        "SM_MODEL_DIR": str(model_directory),
        "MODEL_TYPE": "onnx",
        "OTEL_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
        "CSV_HAS_HEADER": "false",
        "DEFAULT_ACCEPT": "application/json",
    }

    with temporary_environment(environment):
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            readiness_response = await client.get("/ready")
            if readiness_response.status_code != 200:
                raise RuntimeError(
                    f"Readiness failed with status={readiness_response.status_code}"
                )

            invocation_response = await client.post(
                "/invocations",
                content=payload,
                headers={"Content-Type": "text/csv", "Accept": "application/json"},
            )
            if invocation_response.status_code != 200:
                raise RuntimeError(
                    "Invocation failed with "
                    f"status={invocation_response.status_code}, "
                    f"body={invocation_response.text}"
                )
            return cast(list[object], invocation_response.json())


def run_train_and_serve_demo(algorithm_name: str) -> None:
    """Train, export, and serve an ONNX model for one algorithm.

    Parameters
    ----------
    algorithm_name : str
        Model algorithm name.
    """
    model, features = train_model(algorithm_name)
    model_directory = REPOSITORY_ROOT / "tmp" / "examples" / algorithm_name
    model_path = export_model_to_onnx(
        model=model,
        feature_count=features.shape[1],
        output_path=model_directory / "model.onnx",
    )

    predictions = asyncio.run(run_inference_request(model_directory, features[:4]))
    print(f"[{algorithm_name}] model={model_path} predictions={predictions}")
