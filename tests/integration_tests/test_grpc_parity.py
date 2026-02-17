"""Parity checks between HTTP and gRPC serving contracts and behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest

from application import create_application
from config import Settings
from onnx_model_serving.grpc.server import InferenceGrpcService
from parsed_types import ParsedInput

try:
    import grpc

    from onnx_serving_grpc import inference_pb2
except ModuleNotFoundError as exc:
    pytest.skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.integration

_HTTP_TO_GRPC_SURFACE_MAP: dict[str, str] = {
    "/live": "Live",
    "/healthz": "Live",
    "/ready": "Ready",
    "/readyz": "Ready",
    "/ping": "Ready",
    "/invocations": "Predict",
}

_UNMAPPED_HTTP_PATHS: set[str] = set()
_UNMAPPED_GRPC_METHODS: set[str] = set()


class _DummyAdapter:
    """Deterministic adapter used for parity checks."""

    def is_ready(self: _DummyAdapter) -> bool:
        """Return adapter readiness."""
        return True

    def predict(self: _DummyAdapter, parsed_input: ParsedInput) -> object:
        """Return deterministic predictions based on batch size."""
        if parsed_input.X is not None:
            size = int(parsed_input.X.shape[0])
        elif parsed_input.tensors:
            first = next(iter(parsed_input.tensors.values()))
            size = int(first.shape[0]) if first.ndim > 0 else 1
        else:
            size = 1
        return [0] * size


@dataclass
class _RecordingContext:
    """Minimal context mock to capture gRPC error status."""

    code: object | None = None
    details: str | None = None

    def set_code(self: _RecordingContext, code: object) -> None:
        """Capture status code."""
        self.code = code

    def set_details(self: _RecordingContext, details: str) -> None:
        """Capture status details."""
        self.details = details


def _set_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set baseline environment variables for parity tests."""
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("INPUT_MODE", "tabular")
    monkeypatch.setenv("DEFAULT_CONTENT_TYPE", "application/json")
    monkeypatch.setenv("DEFAULT_ACCEPT", "application/json")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("TABULAR_DTYPE", "float32")


def _patch_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch both HTTP and gRPC loaders to use deterministic adapter."""
    import application as application_module
    import onnx_model_serving.grpc.server as grpc_server_module

    monkeypatch.setattr(
        application_module,
        "load_adapter",
        lambda settings: _DummyAdapter(),
    )
    monkeypatch.setattr(
        grpc_server_module,
        "load_adapter",
        lambda settings: _DummyAdapter(),
    )


def _grpc_method_names() -> set[str]:
    """Return gRPC method names from protobuf service descriptor."""
    service = inference_pb2.DESCRIPTOR.services_by_name["InferenceService"]
    return {method.name for method in service.methods}


def _http_route_paths(application: object) -> set[str]:
    """Return documented HTTP paths from OpenAPI schema."""
    app_obj = cast(Any, application)
    return set(app_obj.openapi().get("paths", {}).keys())


def test_surface_parity_http_to_grpc_map_is_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when HTTP/gRPC surfaces diverge without explicit mapping."""
    _set_base_env(monkeypatch)
    _patch_adapters(monkeypatch)
    app = create_application(Settings())

    discovered_http = _http_route_paths(app)
    discovered_grpc = _grpc_method_names()

    mapped_http = set(_HTTP_TO_GRPC_SURFACE_MAP.keys())
    mapped_grpc = set(_HTTP_TO_GRPC_SURFACE_MAP.values())

    assert discovered_http == mapped_http | _UNMAPPED_HTTP_PATHS
    assert discovered_grpc == mapped_grpc | _UNMAPPED_GRPC_METHODS


def test_schema_mapping_predict_fields_are_explicit() -> None:
    """Keep explicit translation table between HTTP and gRPC Predict fields."""
    grpc_predict_request_fields = set(
        inference_pb2.PredictRequest.DESCRIPTOR.fields_by_name.keys()
    )
    grpc_predict_reply_fields = set(
        inference_pb2.PredictReply.DESCRIPTOR.fields_by_name.keys()
    )

    expected_request_mapping = {
        "payload": "HTTP request body bytes",
        "content_type": "HTTP Content-Type header",
        "accept": "HTTP Accept header",
    }
    expected_reply_mapping = {
        "body": "HTTP response body bytes",
        "content_type": "HTTP response Content-Type header",
        "metadata": "HTTP response metadata extension",
    }

    assert grpc_predict_request_fields == set(expected_request_mapping.keys())
    assert grpc_predict_reply_fields == set(expected_reply_mapping.keys())


def test_grpc_contract_for_predict_messages() -> None:
    """Validate gRPC I/O contract fields for predict RPC."""
    request_fields = set(inference_pb2.PredictRequest.DESCRIPTOR.fields_by_name.keys())
    reply_fields = set(inference_pb2.PredictReply.DESCRIPTOR.fields_by_name.keys())
    status_fields = set(inference_pb2.StatusReply.DESCRIPTOR.fields_by_name.keys())

    assert request_fields == {"payload", "content_type", "accept"}
    assert reply_fields == {"body", "content_type", "metadata"}
    assert status_fields == {"ok", "status"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "instances",
    [
        [[1.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]],
    ],
)
async def test_http_and_grpc_predict_parity_across_batches(
    monkeypatch: pytest.MonkeyPatch,
    instances: list[list[float]],
) -> None:
    """Compare HTTP and gRPC outputs for multiple valid payload shapes."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)
    payload_obj = {"instances": instances}
    payload_bytes = json.dumps(payload_obj).encode("utf-8")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        live_response = await client.get("/live")
        healthz_response = await client.get("/healthz")
        ready_response = await client.get("/ready")
        readyz_response = await client.get("/readyz")
        http_response = await client.post(
            "/invocations",
            content=payload_bytes,
            headers={"content-type": "application/json", "accept": "application/json"},
        )

    grpc_live = grpc_service.Live(inference_pb2.LiveRequest(), None)
    grpc_ready = grpc_service.Ready(inference_pb2.ReadyRequest(), None)
    grpc_predict = grpc_service.Predict(
        inference_pb2.PredictRequest(
            payload=payload_bytes,
            content_type="application/json",
            accept="application/json",
        ),
        None,
    )

    assert live_response.status_code == 200
    assert healthz_response.status_code == 200
    assert ready_response.status_code == 200
    assert readyz_response.status_code == 200
    assert http_response.status_code == 200
    assert grpc_live.ok is True
    assert grpc_ready.ok is True
    assert grpc_predict.content_type == "application/json"

    http_json = json.loads(http_response.content.decode("utf-8"))
    grpc_json = json.loads(grpc_predict.body.decode("utf-8"))
    assert http_json == grpc_json


@pytest.mark.asyncio
async def test_error_parity_for_invalid_json_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate invalid JSON maps to equivalent HTTP/gRPC error semantics."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)
    bad_payload = b"{not-json"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/invocations",
            content=bad_payload,
            headers={"content-type": "application/json", "accept": "application/json"},
        )

    context = _RecordingContext()
    grpc_reply = grpc_service.Predict(
        inference_pb2.PredictRequest(
            payload=bad_payload,
            content_type="application/json",
            accept="application/json",
        ),
        context,
    )

    assert http_response.status_code == 400
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert context.details is not None
    assert grpc_reply.body == b""


@pytest.mark.asyncio
async def test_error_parity_for_record_limit_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate batch-limit violations map to equivalent error category."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)
    payload_bytes = json.dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/invocations",
            content=payload_bytes,
            headers={"content-type": "application/json", "accept": "application/json"},
        )

    context = _RecordingContext()
    grpc_service.Predict(
        inference_pb2.PredictRequest(
            payload=payload_bytes,
            content_type="application/json",
            accept="application/json",
        ),
        context,
    )

    assert http_response.status_code == 400
    assert "too_many_records" in http_response.json()["error"]
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert context.details is not None
    assert "too_many_records" in context.details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "content_type", "accept"),
    [
        (b"1,2\n3,4\n", "text/csv", "application/json"),
        (
            json.dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8"),
            "application/json",
            "application/json",
        ),
    ],
)
async def test_http_and_grpc_predict_parity_for_multiple_content_types(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    content_type: str,
    accept: str,
) -> None:
    """Compare HTTP/gRPC outputs for CSV and JSON payload variants."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv("MAX_RECORDS", "1000")
    _patch_adapters(monkeypatch)

    settings = Settings()
    app = create_application(settings)
    grpc_service = InferenceGrpcService(settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        http_response = await client.post(
            "/invocations",
            content=payload,
            headers={"content-type": content_type, "accept": accept},
        )

    grpc_predict = grpc_service.Predict(
        inference_pb2.PredictRequest(
            payload=payload,
            content_type=content_type,
            accept=accept,
        ),
        None,
    )

    assert http_response.status_code == 200
    assert grpc_predict.content_type == "application/json"
    assert json.loads(http_response.content.decode("utf-8")) == json.loads(
        grpc_predict.body.decode("utf-8")
    )
