FROM ffreis/base-builder

USER root

RUN mkdir -p /build \
    && chown appuser:appgroup /build \
    && chmod 0750 /build

WORKDIR /build

COPY --chown=appuser:appgroup pyproject.toml requirements.txt main.py /build/
COPY --chown=appuser:appgroup src /build/src
COPY --chown=appuser:appgroup tests /build/tests
COPY --chown=appuser:appgroup scripts /build/scripts

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install ".[dev]"

# Run tests to ensure everything works
RUN /opt/venv/bin/pytest -q

# Generate lock file for reproducibility
RUN /opt/venv/bin/pip freeze > requirements.lock && chown appuser:appgroup requirements.lock

USER appuser:appgroup

ENTRYPOINT ["/opt/venv/bin/python", "main.py"]
