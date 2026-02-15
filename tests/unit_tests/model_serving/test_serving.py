import pytest

pytestmark = pytest.mark.unit


def test_main_execs_gunicorn(monkeypatch):
    import serving as serving_module

    seen = {}

    def fake_execvp(cmd, argv):
        seen["cmd"] = cmd
        seen["argv"] = argv
        raise RuntimeError("stop")

    monkeypatch.setattr(serving_module.os, "execvp", fake_execvp)
    with pytest.raises(RuntimeError, match="stop"):
        serving_module.main()

    assert seen["cmd"] == "gunicorn"
    assert seen["argv"] == [
        "gunicorn",
        "-c",
        "python:gunicorn_configuration",
        "serving:application",
    ]


def test_module_exposes_asgi_app():
    import serving as serving_module

    assert serving_module.application is not None
