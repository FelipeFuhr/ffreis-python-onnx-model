import importlib

import pytest

pytestmark = pytest.mark.unit


def test_gunicorn_configuration_uses_settings_from_env(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("GUNICORN_WORKERS", "3")
    monkeypatch.setenv("GUNICORN_THREADS", "9")
    monkeypatch.setenv("GUNICORN_TIMEOUT", "88")
    monkeypatch.setenv("GUNICORN_GRACEFUL_TIMEOUT", "44")
    monkeypatch.setenv("GUNICORN_KEEPALIVE", "7")

    import gunicorn_configuration as gunicorn_configuration

    gunicorn_settings_module = importlib.reload(gunicorn_configuration)
    assert gunicorn_settings_module.bind == "0.0.0.0:9999"
    assert gunicorn_settings_module.workers == 3
    assert gunicorn_settings_module.threads == 9
    assert gunicorn_settings_module.timeout == 88
    assert gunicorn_settings_module.graceful_timeout == 44
    assert gunicorn_settings_module.keepalive == 7
    assert gunicorn_settings_module.worker_class == "uvicorn.workers.UvicornWorker"
    assert gunicorn_settings_module.accesslog == "-"
    assert gunicorn_settings_module.errorlog == "-"
