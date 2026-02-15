"""Telemetry setup and instrumentation helpers."""

from __future__ import annotations

import importlib
import logging
from typing import Protocol, cast

from fastapi import FastAPI

from config import Settings

log = logging.getLogger("otel")

trace: object | None = None
OTLPSpanExporter: object | None = None
FastAPIInstrumentor: object | None = None
LoggingInstrumentor: object | None = None
RequestsInstrumentor: object | None = None
Resource: object | None = None
TracerProvider: object | None = None
BatchSpanProcessor: object | None = None


def _load_optional_telemetry_components() -> None:
    """Load optional telemetry dependencies if available."""
    global trace
    global OTLPSpanExporter
    global FastAPIInstrumentor
    global LoggingInstrumentor
    global RequestsInstrumentor
    global Resource
    global TracerProvider
    global BatchSpanProcessor

    try:
        trace = importlib.import_module("opentelemetry").trace
        OTLPSpanExporter = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ).OTLPSpanExporter
        FastAPIInstrumentor = importlib.import_module(
            "opentelemetry.instrumentation.fastapi"
        ).FastAPIInstrumentor
        LoggingInstrumentor = importlib.import_module(
            "opentelemetry.instrumentation.logging"
        ).LoggingInstrumentor
        RequestsInstrumentor = importlib.import_module(
            "opentelemetry.instrumentation.requests"
        ).RequestsInstrumentor
        Resource = importlib.import_module("opentelemetry.sdk.resources").Resource
        TracerProvider = importlib.import_module(
            "opentelemetry.sdk.trace"
        ).TracerProvider
        BatchSpanProcessor = importlib.import_module(
            "opentelemetry.sdk.trace.export"
        ).BatchSpanProcessor
    except Exception:  # pragma: no cover - optional dependency
        trace = None
        OTLPSpanExporter = None
        FastAPIInstrumentor = None
        LoggingInstrumentor = None
        RequestsInstrumentor = None
        Resource = None
        TracerProvider = None
        BatchSpanProcessor = None


_load_optional_telemetry_components()


class TracerProtocol(Protocol):
    """Protocol for OpenTelemetry tracer-like objects."""

    def start_as_current_span(self: TracerProtocol, name: str) -> object:
        """Start span context manager."""


class TraceModuleProtocol(Protocol):
    """Protocol for `opentelemetry.trace` module API used here."""

    def get_tracer(self: TraceModuleProtocol, name: str) -> TracerProtocol:
        """Return tracer by name."""

    def set_tracer_provider(self: TraceModuleProtocol, provider: object) -> None:
        """Set tracer provider."""


class ResourceProtocol(Protocol):
    """Protocol for OpenTelemetry resource factory."""

    @staticmethod
    def create(payload: dict[str, str]) -> object:
        """Create resource object."""


class TracerProviderProtocol(Protocol):
    """Protocol for trace provider constructor and methods."""

    def add_span_processor(self: TracerProviderProtocol, processor: object) -> None:
        """Attach span processor."""


class TracerProviderFactoryProtocol(Protocol):
    """Protocol for trace provider constructor."""

    def __call__(
        self: TracerProviderFactoryProtocol, *, resource: object
    ) -> TracerProviderProtocol:
        """Create provider instance."""


class ExporterFactoryProtocol(Protocol):
    """Protocol for exporter constructor."""

    def __call__(
        self: ExporterFactoryProtocol,
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
    ) -> object:
        """Create exporter."""


class BatchProcessorFactoryProtocol(Protocol):
    """Protocol for batch span processor constructor."""

    def __call__(self: BatchProcessorFactoryProtocol, exporter: object) -> object:
        """Create batch span processor."""


class InstrumentorFactoryProtocol(Protocol):
    """Protocol for instrumentation factories."""

    def __call__(self: InstrumentorFactoryProtocol) -> InstrumentorProtocol:
        """Create instrumentor instance."""


class InstrumentorProtocol(Protocol):
    """Protocol for request/logging instrumentors."""

    def instrument(
        self: InstrumentorProtocol, set_logging_format: bool = False
    ) -> None:
        """Enable instrumentation."""


class FastApiInstrumentorProtocol(Protocol):
    """Protocol for FastAPI instrumentor."""

    @staticmethod
    def instrument_app(application: FastAPI) -> None:
        """Instrument FastAPI application."""


def get_tracer(name: str) -> TracerProtocol | None:
    """Return configured tracer when telemetry dependency is present."""
    if trace is None:
        return None
    trace_module = cast(TraceModuleProtocol, trace)
    return trace_module.get_tracer(name)


def _parse_headers(s: str) -> dict[str, str]:
    """Parse OTLP headers from comma-separated key-value string.

    Parameters
    ----------
    s : str
        Header string formatted as ``k1=v1,k2=v2``.

    Returns
    -------
    dict[str, str]
        Parsed header mapping.
    """
    parsed_headers = {}
    for part in [p.strip() for p in (s or "").split(",") if p.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            parsed_headers[k.strip()] = v.strip()
    return parsed_headers


def setup_telemetry(settings: Settings) -> bool:
    """Configure OpenTelemetry exporters and instrumentors.

    Parameters
    ----------
    settings : Settings
        Runtime settings containing telemetry controls.

    Returns
    -------
    bool
        ``True`` when telemetry exporter has been configured.
    """
    if not settings.otel_enabled:
        log.info("OTEL disabled")
        return False

    if (
        trace is None
        or Resource is None
        or TracerProvider is None
        or OTLPSpanExporter is None
        or BatchSpanProcessor is None
        or RequestsInstrumentor is None
        or LoggingInstrumentor is None
    ):
        log.info("OTEL dependencies unavailable; exporter disabled.")
        return False

    if not settings.otel_endpoint:
        log.info("OTEL enabled but no endpoint configured; exporter disabled.")
        return False

    resource = cast(ResourceProtocol, Resource).create(
        {
            "service.name": settings.service_name,
            "service.version": settings.service_version,
            "deployment.environment": settings.deployment_env,
            "cloud.provider": "aws",
            "cloud.platform": "aws_sagemaker",
        }
    )

    provider = cast(TracerProviderFactoryProtocol, TracerProvider)(resource=resource)
    cast(TraceModuleProtocol, trace).set_tracer_provider(provider)

    exporter = cast(ExporterFactoryProtocol, OTLPSpanExporter)(
        endpoint=settings.otel_endpoint,
        headers=_parse_headers(settings.otel_headers),
        timeout=settings.otel_timeout_s,
    )
    provider.add_span_processor(
        cast(BatchProcessorFactoryProtocol, BatchSpanProcessor)(exporter)
    )

    cast(InstrumentorFactoryProtocol, RequestsInstrumentor)().instrument()
    cast(InstrumentorFactoryProtocol, LoggingInstrumentor)().instrument(
        set_logging_format=True
    )

    log.info("OTEL exporter configured: %s", settings.otel_endpoint)
    return True


def instrument_fastapi_application(settings: Settings, application: FastAPI) -> None:
    """Instrument a FastAPI application when telemetry is enabled.

    Parameters
    ----------
    settings : Settings
        Runtime settings containing telemetry controls.
    application : fastapi.FastAPI
        FastAPI application instance.
    """
    if settings.otel_enabled and FastAPIInstrumentor is not None:
        cast(FastApiInstrumentorProtocol, FastAPIInstrumentor).instrument_app(
            application
        )
