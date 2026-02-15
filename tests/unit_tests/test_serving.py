"""Tests for serving."""

import pytest

pytestmark = pytest.mark.unit


def test_main_execs_gunicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main execs gunicorn.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to configure environment and runtime hooks.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    import serving as serving_module

    seen = {}

    def fake_execvp(cmd: object, argv: object) -> object:
        """Run fake execvp.

        Parameters
        ----------
        cmd : object
            Parameter used by this test scenario.
        argv : object
            Parameter used by this test scenario.

        Returns
        -------
        object
            Return value produced by helper logic in this test module.
        """
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


def test_module_exposes_asgi_app() -> None:
    """Verify module exposes asgi app.

    Returns
    -------
    None
        Does not return a value; assertions validate expected behavior.
    """
    import serving as serving_module

    assert serving_module.application is not None
