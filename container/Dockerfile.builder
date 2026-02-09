FROM ffreis/base-builder

USER root

RUN mkdir -p /build \
    && chown appuser:appgroup /build \
    && chmod 0750 /build

WORKDIR /build

COPY --chown=appuser:appgroup app/ .

# Install dependencies as root to avoid permission issues with Python 3.13
RUN pip install --break-system-packages -e ".[dev]"

# Run tests to ensure everything works
RUN pytest -v

# Generate lock file for reproducibility
RUN pip freeze > requirements.lock && chown appuser:appgroup requirements.lock

USER appuser:appgroup

ENTRYPOINT ["python3", "main.py"]
