"""Telemetry setup and instrumentation helpers."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from config import Settings

log = logging.getLogger("otel")

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    OTLPSpanExporter = None
    FastAPIInstrumentor = None
    LoggingInstrumentor = None
    RequestsInstrumentor = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None


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

    if trace is None:
        log.info("OTEL dependencies unavailable; exporter disabled.")
        return False

    if not settings.otel_endpoint:
        log.info("OTEL enabled but no endpoint configured; exporter disabled.")
        return False

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.service_version,
            "deployment.environment": settings.deployment_env,
            "cloud.provider": "aws",
            "cloud.platform": "aws_sagemaker",
        }
    )

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(
        endpoint=settings.otel_endpoint,
        headers=_parse_headers(settings.otel_headers),
        timeout=settings.otel_timeout_s,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=True)

    log.info("OTEL exporter configured: %s", settings.otel_endpoint)
    return True


def instrument_fastapi_application(settings: Settings, application: FastAPI) -> None:
    """Instrument a FastAPI application when telemetry is enabled.

    Parameters
    ----------
    settings : Settings
        Runtime settings containing telemetry controls.
    application : Any
        FastAPI application instance.
    """
    if settings.otel_enabled and FastAPIInstrumentor is not None:
        FastAPIInstrumentor.instrument_app(application)
