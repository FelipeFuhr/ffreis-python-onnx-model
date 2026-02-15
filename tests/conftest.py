import httpx
import pytest
import pytest_asyncio

from application import create_application
from config import Settings


class DummyAdapter:
    def __init__(self, mode="list"):
        self.mode = mode

    def is_ready(self):
        return True

    def predict(self, parsed_input):
        if self.mode == "list":
            if parsed_input.X is not None:
                n = parsed_input.X.shape[0]
            elif parsed_input.tensors:
                n = next(iter(parsed_input.tensors.values())).shape[0]
            else:
                n = 1
            return [0] * int(n)
        if self.mode == "dict":
            return {"logits": [[1.0, 2.0]], "proba": [[0.1, 0.9]]}
        raise ValueError("Unknown mode")


@pytest.fixture
def base_env(monkeypatch):
    monkeypatch.setenv("INPUT_MODE", "tabular")
    monkeypatch.setenv("DEFAULT_CONTENT_TYPE", "application/json")
    monkeypatch.setenv("DEFAULT_ACCEPT", "application/json")
    monkeypatch.setenv("CSV_DELIMITER", ",")
    monkeypatch.setenv("CSV_HAS_HEADER", "false")
    monkeypatch.setenv("TABULAR_DTYPE", "float32")
    monkeypatch.setenv("TABULAR_NUM_FEATURES", "0")
    monkeypatch.setenv("MAX_BODY_BYTES", "1000000")
    monkeypatch.setenv("MAX_RECORDS", "1000")
    monkeypatch.setenv("MAX_INFLIGHT", "4")
    monkeypatch.setenv("ACQUIRE_TIMEOUT_S", "0.2")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")
    monkeypatch.setenv("PROMETHEUS_PATH", "/metrics")
    monkeypatch.setenv("OTEL_ENABLED", "false")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    monkeypatch.setenv("MODEL_TYPE", "onnx")
    monkeypatch.setenv("SM_MODEL_DIR", "/opt/ml/model")
    yield


def _make_client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def app_list_adapter(monkeypatch, base_env):
    import application as application_module

    monkeypatch.setattr(
        application_module, "load_adapter", lambda settings: DummyAdapter(mode="list")
    )
    return create_application(Settings())


@pytest.fixture
def app_dict_adapter(monkeypatch, base_env):
    import application as application_module

    monkeypatch.setattr(
        application_module, "load_adapter", lambda settings: DummyAdapter(mode="dict")
    )
    return create_application(Settings())


@pytest_asyncio.fixture
async def client_list(app_list_adapter):
    async with _make_client(app_list_adapter) as client:
        yield client


@pytest_asyncio.fixture
async def client_dict(app_dict_adapter):
    async with _make_client(app_dict_adapter) as client:
        yield client
