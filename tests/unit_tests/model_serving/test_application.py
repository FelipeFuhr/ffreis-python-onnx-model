import httpx
import pytest

from application import create_application
from config import Settings

pytestmark = pytest.mark.unit


class TestAppEndpoints:
    @pytest.mark.asyncio
    async def test_live_is_process_only_healthcheck(self, client_list):
        response = await client_list.get("/live")
        assert response.status_code == 200
        assert response.text.strip() == ""

    @pytest.mark.asyncio
    async def test_ready_reports_model_readiness(self, client_list):
        response = await client_list.get("/ready")
        assert response.status_code == 200
        assert response.text.strip() == ""

    @pytest.mark.asyncio
    async def test_ping_ok(self, client_list):
        r = await client_list.get("/ping")
        assert r.status_code == 200
        assert r.text.strip() == ""

    @pytest.mark.asyncio
    async def test_ping_is_alias_for_ready_when_not_ready(self, monkeypatch):
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
    async def test_metrics_exists(self, client_list):
        r = await client_list.get("/metrics")
        assert r.status_code == 200
        assert "# HELP" in r.text or "# TYPE" in r.text

    @pytest.mark.asyncio
    async def test_ping_returns_500_when_adapter_fails(self, monkeypatch):
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
    async def test_invocations_csv_basic(self, client_list):
        r = await client_list.post(
            "/invocations",
            content=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == [0, 0]

    @pytest.mark.asyncio
    async def test_invocations_respects_max_body_bytes(self, monkeypatch):
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
    async def test_invocations_respects_max_records(self, monkeypatch):
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
    async def test_sagemaker_header_fallback_content_type_accept(self, client_list):
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
    async def test_dict_output_forces_json_even_if_accept_csv(self, client_dict):
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
    async def test_invocations_returns_400_for_bad_payload(self, monkeypatch):
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
    async def test_invocations_returns_500_for_adapter_exception(self, monkeypatch):
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
    async def test_invocations_returns_429_when_inflight_exhausted(self, monkeypatch):
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
