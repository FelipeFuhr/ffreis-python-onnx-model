"""Tests for application."""

import types
from typing import Self

import httpx
import numpy as np
import pytest

from application import _batch_size, _NoopSpan, create_application
from config import Settings
from parsed_types import ParsedInput

pytestmark = pytest.mark.unit


def test_noop_span_context_manager_contract() -> None:
    """Verify no-op span enter/exit behavior."""
    span = _NoopSpan()
    assert span.__enter__() is span
    assert span.__exit__(None, None, None) is False
    assert span.set_attribute("k", "v") is None


def test_batch_size_uses_tensors_and_raises_for_empty_payload() -> None:
    """Verify batch-size extraction from tensors and empty payload errors."""
    parsed_tensors = ParsedInput(tensors={"x": np.zeros((2, 3), dtype=np.float32)})
    assert _batch_size(parsed_tensors) == 2

    parsed_scalar_tensor = ParsedInput(tensors={"x": np.array(1.0, dtype=np.float32)})
    assert _batch_size(parsed_scalar_tensor) == 1

    with pytest.raises(ValueError, match="no features/tensors"):
        _batch_size(ParsedInput())


def test_builder_configures_telemetry_when_setup_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify telemetry instrumentation is wired when setup reports enabled."""
    import application as application_module

    instrument_calls: list[object] = []
    monkeypatch.setattr(application_module, "setup_telemetry", lambda settings: True)
    monkeypatch.setattr(
        application_module,
        "instrument_fastapi_application",
        lambda settings, application: instrument_calls.append(application),
    )

    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    create_application(Settings())
    assert instrument_calls


@pytest.mark.asyncio
async def test_builder_run_inference_raises_when_adapter_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise runtime error when adapter loading leaves adapter unset."""
    import application as application_module

    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    builder = application_module.InferenceApplicationBuilder(Settings())
    builder._ensure_adapter_loaded = types.MethodType(
        lambda self: None,
        builder,
    )
    with pytest.raises(RuntimeError, match="Adapter failed to load"):
        await builder._run_inference(  # noqa: SLF001
            request=object(),  # type: ignore[arg-type]
            start_time=0.0,
        )


def test_readiness_returns_500_when_adapter_stays_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return 500 readiness when loader runs but adapter remains unavailable."""
    import application as application_module

    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    builder = application_module.InferenceApplicationBuilder(Settings())
    builder._ensure_adapter_loaded = types.MethodType(
        lambda self: None,
        builder,
    )
    response = builder._build_readiness_response()  # noqa: SLF001
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_metrics_fallback_route_when_instrumentator_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose fallback metrics route when instrumentator dependency is missing."""
    import application as application_module

    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")
    monkeypatch.setattr(application_module, "Instrumentator", None)
    application = create_application(Settings())
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "byoc_up 1" in response.text


class TestAppEndpoints:
    """Test suite for TestAppEndpoints."""

    @pytest.mark.asyncio
    async def test_live_is_process_only_healthcheck(
        self: Self, client_list: httpx.AsyncClient
    ) -> None:
        """Verify live is process only healthcheck.

        Parameters
        ----------
        client_list : httpx.AsyncClient
            Async HTTP client fixture targeting the list-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        response = await client_list.get("/live")
        assert response.status_code == 200
        assert response.text.strip() == ""

    @pytest.mark.asyncio
    async def test_ready_reports_model_readiness(
        self: Self, client_list: httpx.AsyncClient
    ) -> None:
        """Verify ready reports model readiness.

        Parameters
        ----------
        client_list : httpx.AsyncClient
            Async HTTP client fixture targeting the list-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        response = await client_list.get("/ready")
        assert response.status_code == 200
        assert response.text.strip() == ""

    @pytest.mark.asyncio
    async def test_ping_ok(self: Self, client_list: httpx.AsyncClient) -> None:
        """Verify ping ok.

        Parameters
        ----------
        client_list : httpx.AsyncClient
            Async HTTP client fixture targeting the list-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        r = await client_list.get("/ping")
        assert r.status_code == 200
        assert r.text.strip() == ""

    @pytest.mark.asyncio
    async def test_ping_is_alias_for_ready_when_not_ready(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate ping is alias for ready when not ready.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {
                    "is_ready": lambda self: False,
                    "predict": lambda self, parsed_input: [0],
                },
            )(),
        )
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            ready_response = await client.get("/ready")
            ping_response = await client.get("/ping")
            assert ready_response.status_code == 500
            assert ping_response.status_code == 500

    @pytest.mark.asyncio
    async def test_metrics_exists(self: Self, client_list: httpx.AsyncClient) -> None:
        """Verify metrics exists.

        Parameters
        ----------
        client_list : httpx.AsyncClient
            Async HTTP client fixture targeting the list-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        r = await client_list.get("/metrics")
        assert r.status_code == 200
        assert "# HELP" in r.text or "# TYPE" in r.text

    @pytest.mark.asyncio
    async def test_ping_returns_500_when_adapter_fails(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate ping returns 500 when adapter fails.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.get("/ping")
            assert r.status_code == 500

    @pytest.mark.asyncio
    async def test_invocations_csv_basic(
        self: Self, client_list: httpx.AsyncClient
    ) -> None:
        """Verify invocations csv basic.

        Parameters
        ----------
        client_list : httpx.AsyncClient
            Async HTTP client fixture targeting the list-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        r = await client_list.post(
            "/invocations",
            content=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == [0, 0]

    @pytest.mark.asyncio
    async def test_invocations_include_trace_correlation_headers(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify invocation responses include trace correlation headers."""
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {"is_ready": lambda self: True, "predict": lambda self, inp: [0, 0]},
            )(),
        )
        monkeypatch.setattr(
            application_module,
            "get_current_trace_identifiers",
            lambda: {"trace_id": "a" * 32, "span_id": "b" * 16},
        )
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/invocations",
                content=b"1,2,3\n4,5,6\n",
                headers={"Content-Type": "text/csv", "Accept": "application/json"},
            )
            assert response.status_code == 200
            assert response.headers.get("x-trace-id") == "a" * 32
            assert response.headers.get("x-span-id") == "b" * 16

    @pytest.mark.asyncio
    async def test_invocations_respects_max_body_bytes(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate invocations respects max body bytes.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MAX_BODY_BYTES", "10")
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {"is_ready": lambda self: True, "predict": lambda self, inp: [0]},
            )(),
        )
        application = create_application(Settings())

        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.post(
                "/invocations",
                content=b"1,2,3\n4,5,6\n",
                headers={"Content-Type": "text/csv"},
            )
            assert r.status_code == 413
            assert r.json()["error"] == "payload_too_large"

    @pytest.mark.asyncio
    async def test_invocations_respects_max_records(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify invocations respects max records.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MAX_RECORDS", "1")
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {"is_ready": lambda self: True, "predict": lambda self, inp: [0]},
            )(),
        )
        application = create_application(Settings())

        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.post(
                "/invocations",
                content=b"1,2,3\n4,5,6\n",
                headers={"Content-Type": "text/csv"},
            )
            assert r.status_code == 400
            assert "too_many_records" in r.json()["error"]

    @pytest.mark.asyncio
    async def test_sagemaker_header_fallback_content_type_accept(
        self: Self, client_list: httpx.AsyncClient
    ) -> None:
        """Validate sagemaker header fallback content type accept.

        Parameters
        ----------
        client_list : httpx.AsyncClient
            Async HTTP client fixture targeting the list-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        r = await client_list.post(
            "/invocations",
            content=b"1,2,3\n",
            headers={
                "X-Amzn-SageMaker-Content-Type": "text/csv",
                "X-Amzn-SageMaker-Accept": "application/json",
            },
        )
        assert r.status_code == 200
        assert r.json() == [0]

    @pytest.mark.asyncio
    async def test_dict_output_forces_json_even_if_accept_csv(
        self: Self, client_dict: httpx.AsyncClient
    ) -> None:
        """Validate dict output forces json even if accept csv.

        Parameters
        ----------
        client_dict : httpx.AsyncClient
            Async HTTP client fixture targeting the dictionary-adapter application.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        r = await client_dict.post(
            "/invocations",
            content=b"1,2,3\n",
            headers={"Content-Type": "text/csv", "Accept": "text/csv"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert "logits" in body and "proba" in body

    @pytest.mark.asyncio
    async def test_invocations_returns_400_for_bad_payload(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate invocations returns 400 for bad payload.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {"is_ready": lambda self: True, "predict": lambda self, inp: [0]},
            )(),
        )
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.post(
                "/invocations",
                content=b"<bad/>",
                headers={"Content-Type": "application/xml"},
            )
            assert r.status_code == 400
            assert "Unsupported Content-Type" in r.json()["error"]

    @pytest.mark.asyncio
    async def test_invocations_returns_500_for_adapter_exception(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate invocations returns 500 for adapter exception.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {
                    "is_ready": lambda self: True,
                    "predict": lambda self, inp: (_ for _ in ()).throw(
                        RuntimeError("boom")
                    ),
                },
            )(),
        )
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.post(
                "/invocations",
                content=b"1,2,3\n",
                headers={"Content-Type": "text/csv"},
            )
            assert r.status_code == 500
            assert r.json()["error"] == "internal_server_error"

    @pytest.mark.asyncio
    async def test_invocations_returns_429_when_inflight_exhausted(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate invocations returns 429 when inflight exhausted.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture used to configure environment and runtime hooks.

        Returns
        -------
        None
            Does not return a value; assertions validate expected behavior.
        """
        monkeypatch.setenv("MAX_INFLIGHT", "0")
        monkeypatch.setenv("ACQUIRE_TIMEOUT_S", "0.001")
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
        import application as application_module

        monkeypatch.setattr(
            application_module,
            "load_adapter",
            lambda settings: type(
                "A",
                (object,),
                {"is_ready": lambda self: True, "predict": lambda self, inp: [0]},
            )(),
        )
        application = create_application(Settings())
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.post(
                "/invocations",
                content=b"1,2,3\n",
                headers={"Content-Type": "text/csv"},
            )
            assert r.status_code == 429
            assert r.json()["error"] == "too_many_requests"
