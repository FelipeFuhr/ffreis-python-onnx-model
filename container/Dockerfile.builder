ARG BASE_BUILDER_IMAGE=ffreis/base-builder
ARG UV_VENV_IMAGE=ffreis/uv-venv
FROM ${BASE_BUILDER_IMAGE}
ARG UV_VENV_IMAGE

USER root

RUN mkdir -p /build \
    && chown appuser:appgroup /build \
    && chmod 0750 /build

WORKDIR /build

COPY --from=${UV_VENV_IMAGE} --chown=appuser:appgroup /opt/venv /opt/venv
COPY --chown=appuser:appgroup pyproject.toml uv.lock main.py /build/
COPY --chown=appuser:appgroup docs /build/docs
COPY --chown=appuser:appgroup src /build/src
COPY --chown=appuser:appgroup tests /build/tests
COPY --chown=appuser:appgroup scripts /build/scripts

# Run tests to ensure everything works
RUN . /opt/venv/bin/activate && uv run --active pytest -q

USER appuser:appgroup

ENTRYPOINT ["/opt/venv/bin/python", "main.py"]
