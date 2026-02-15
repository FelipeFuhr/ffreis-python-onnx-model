import types

import pytest

from config import Settings
from telemetry import (
    _parse_headers,
    instrument_fastapi_application,
    setup_telemetry,
)

pytestmark = pytest.mark.unit


class _Recorder:
    def __init__(self):
        self.calls = []

    def add(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_parse_headers_handles_empty_and_pairs():
    assert _parse_headers("") == {}
    assert _parse_headers("a=b, c=d ,bad, x=1=2") == {"a": "b", "c": "d", "x": "1=2"}


def test_setup_telemetry_returns_false_when_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "false")
    assert setup_telemetry(Settings()) is False


def test_setup_telemetry_returns_false_when_dependencies_missing(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")
    import telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "trace", None)
    assert setup_telemetry(Settings()) is False


def test_setup_telemetry_returns_false_without_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    import telemetry as telemetry_module

    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types.SimpleNamespace(set_tracer_provider=lambda p: None),
    )
    assert setup_telemetry(Settings()) is False


def test_setup_telemetry_happy_path(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "a=b")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "3")

    import telemetry as telemetry_module

    trace_rec = _Recorder()
    req_rec = _Recorder()
    log_rec = _Recorder()
    resource_rec = _Recorder()
    provider_rec = _Recorder()

    class FakeProvider:
        def __init__(self, resource):
            provider_rec.add(resource=resource)

        def add_span_processor(self, processor):
            provider_rec.add(processor=processor)

    class FakeResource:
        @staticmethod
        def create(payload):
            resource_rec.add(payload=payload)
            return {"resource": payload}

    class FakeExporter:
        def __init__(self, endpoint, headers, timeout):
            self.endpoint = endpoint
            self.headers = headers
            self.timeout = timeout

    class FakeBatch:
        def __init__(self, exporter):
            self.exporter = exporter

    monkeypatch.setattr(
        telemetry_module,
        "trace",
        types.SimpleNamespace(
            set_tracer_provider=lambda provider: trace_rec.add(provider=provider)
        ),
    )
    monkeypatch.setattr(telemetry_module, "Resource", FakeResource)
    monkeypatch.setattr(telemetry_module, "TracerProvider", FakeProvider)
    monkeypatch.setattr(telemetry_module, "OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr(telemetry_module, "BatchSpanProcessor", FakeBatch)
    monkeypatch.setattr(
        telemetry_module,
        "RequestsInstrumentor",
        lambda: types.SimpleNamespace(instrument=lambda: req_rec.add()),
    )
    monkeypatch.setattr(
        telemetry_module,
        "LoggingInstrumentor",
        lambda: types.SimpleNamespace(
            instrument=lambda set_logging_format: log_rec.add(
                set_logging_format=set_logging_format
            )
        ),
    )

    out = setup_telemetry(Settings())
    assert out is True
    assert trace_rec.calls
    assert resource_rec.calls
    assert provider_rec.calls
    assert req_rec.calls
    assert log_rec.calls


def test_instrument_fastapi_when_enabled(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")
    import telemetry as telemetry_module

    rec = _Recorder()
    monkeypatch.setattr(
        telemetry_module,
        "FastAPIInstrumentor",
        types.SimpleNamespace(
            instrument_app=lambda application: rec.add(application=application)
        ),
    )
    application = object()
    instrument_fastapi_application(Settings(), application)
    assert rec.calls


def test_instrument_fastapi_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "false")
    import telemetry as telemetry_module

    rec = _Recorder()
    monkeypatch.setattr(
        telemetry_module,
        "FastAPIInstrumentor",
        types.SimpleNamespace(
            instrument_app=lambda application: rec.add(application=application)
        ),
    )
    instrument_fastapi_application(Settings(), object())
    assert rec.calls == []
