"""Shared utilities for training and serving ONNX example models."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application import create_application  # noqa: E402
from config import Settings  # noqa: E402


def train_model(algorithm_name: str) -> tuple[BaseEstimator, np.ndarray]:
    """Train a sklearn model on Iris and return model plus features.

    Parameters
    ----------
    algorithm_name : str
        Model algorithm name. One of ``logistic_regression``, ``random_forest``,
        or ``neural_network``.

    Returns
    -------
    tuple[BaseEstimator, numpy.ndarray]
        Trained model and full training feature matrix.
    """
    iris = load_iris()
    features = iris.data.astype(np.float32)
    labels = iris.target

    if algorithm_name == "logistic_regression":
        model = LogisticRegression(max_iter=300)
    elif algorithm_name == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif algorithm_name == "neural_network":
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=600,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    model.fit(features, labels)
    return model, features


def export_model_to_onnx(
    model: BaseEstimator, feature_count: int, output_path: Path
) -> Path:
    """Export a sklearn model to ONNX.

    Parameters
    ----------
    model : BaseEstimator
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("x", FloatTensorType([None, feature_count]))]
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
            return invocation_response.json()


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
