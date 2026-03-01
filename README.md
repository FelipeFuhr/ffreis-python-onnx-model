# Incremental Docker Build with Python

A minimal guide to building Python applications with Docker using multi-stage builds for lightweight container images with reproducible builds and testing.

[![Docker Build](https://github.com/FelipeFuhr/ffreis-python-onnx-model/actions/workflows/docker-build.yml/badge.svg?branch=main)](https://github.com/FelipeFuhr/ffreis-python-onnx-model/actions/workflows/docker-build.yml)

## What is this?

This project demonstrates a **multi-stage Docker build** for Python that:

1. **Stage 1 (Builder)**: Creates an isolated virtual environment, installs dependencies from `uv.lock`, and runs tests
2. **Stage 2 (Runtime)**: Copies only the necessary application files plus the built virtual environment to a minimal image

## API Contract

- OpenAPI transport contract: `docs/openapi.yaml`
- gRPC contract: `proto/onnx_serving_grpc/inference.proto`
  - Server reflection is intentionally not enabled in runtime paths.

OpenAPI documents transport behavior (paths, media types, headers, and response envelopes).
Model-specific tensor semantics are expected to be defined by a model manifest shipped with artifacts.

## Quick Start

### Build all images

```bash
make build-images
```

### Build only the builder (includes testing)

```bash
make build-builder
```

### Run the app container

```bash
docker run ffreis/runner
```

## How it works

This project uses an **incremental multi-image approach** optimized for Python development:

**Image Layers:**

1. **Base (`ffreis/base`)**: Lightweight Ubuntu 26.04 base image with unprivileged user
2. **Base Builder (`ffreis/base-builder`)**: Adds Python and virtualenv tooling to the base
3. **UV Venv (`ffreis/uv-venv`)**: Builds shared `/opt/venv` from `uv.lock`
4. **Builder (`ffreis/builder`)**: Reuses `/opt/venv` and runs tests
5. **Base Runner (`ffreis/base-runner`)**: Minimal runtime base with entrypoint script
6. **Runner (`ffreis/runner`)**: Contains application code, Python runtime, and copied `/opt/venv`

**Benefits:**

- **Reproducible builds**: `uv.lock` ensures consistent dependencies across environments
- **Testing in build**: Tests run during the build process - build fails if tests fail
- **Layer caching**: Rebuild only changed layers; base images rarely change
- **Minimal final images**: Runtime excludes build tools, tests, and unnecessary files
- **Efficient builds**: Parallel caching speeds up repeated builds
- **Security**: Runs as unprivileged user, minimal attack surface

## Available Commands

### Build targets

```bash
make build-base              # Build base Ubuntu image
make build-base-builder      # Build base image with Python
make build-uv-venv           # Build shared uv-based virtual environment image
make build-builder           # Build builder (reuses uv-venv and runs tests)
make build-base-runner       # Build minimal runner base
make build-runner            # Build final runner image
make build-images            # Build all images at once
```

### Run targets

```bash
make run                     # Run app locally
make run-container           # Run the app in runner container
```

### Cleanup targets

```bash
make clean-base              # Remove base image
make clean-base-builder      # Remove base-builder image
make clean-base-runner       # Remove base-runner image
make clean-runner            # Remove runner image
make clean-all               # Remove all images
```

## Why this approach?

- **Reproducibility**: Lock files ensure consistent builds across environments
- **Quality gates**: Tests run during build, preventing broken images from being created
- **Incremental builds**: Cache base images separately; only rebuild changed layers
- **Smaller final images**: Runtime excludes all build tools, tests, and dev dependencies
- **Reusable components**: Share `ffreis/base` and Python builder across projects
- **Faster CI/CD**: Docker layer caching speeds up repeated builds
- **Security**: Minimize attack surface by running as unprivileged user and shipping only necessary files
- **Flexibility**: Easy to swap Python versions or base images

## Project Structure

```
.
├── main.py                 # Application entry point
├── pyproject.toml          # Python project configuration & dependencies
├── uv.lock                 # Locked dependency graph used by uv
├── src/
│   └── onnx_model_serving/
│       ├── __init__.py
│       └── lib.py
├── tests/
│   ├── unit_tests/
│   ├── integration_tests/
│   └── e2e_tests/
├── container/             # Docker multi-stage build files
│   ├── digests.env       # Base image digest pinning
│   ├── Dockerfile.base
│   ├── Dockerfile.base-builder
│   ├── Dockerfile.uv-builder # Builds /opt/venv from uv.lock
│   ├── Dockerfile.builder  # Installs deps from uv.lock and runs tests
│   ├── Dockerfile.base-runner
│   └── Dockerfile.runner
├── scripts/              # Helper scripts
│   └── entrypoint.sh    # Container entrypoint
├── Makefile             # Main build orchestration
├── .pre-commit-config.yaml
├── .deepsource.toml
├── sonar-project.properties
└── .github/
    └── workflows/
        ├── build-all.yml
        ├── code-quality.yml
        ├── code-review.yml
        ├── commit-checks.yml
        ├── coverage.yml
        ├── deepsource.yml
        ├── docker-build.yml
        ├── e2e.yml
        ├── grype.yml
        ├── integration.yml
        ├── lint.yml
        └── trivy.yml

## Development Workflow

1. **Add dependencies**: Update `pyproject.toml` with new dependencies
2. **Run locally**:
   - `make env`
   - `make build-local`
   - `make run`
3. **Build images**: Run `make build-images` to build and test in containers
4. **Tests run automatically**: Builder stage runs `pytest` - build fails if tests fail
5. **Locked install enforced**: `uv.lock` is used with `--frozen` for reproducible deployments
6. **Deploy**: Use `ffreis/runner` image in production - it's minimal and secure

## Testing

Tests are split by scope in `tests/unit_tests`, `tests/integration_tests`, and `tests/e2e_tests`. They run automatically during Docker builder image creation.

```bash
# Run tests locally
make env
make build-local
make test-unit
make test-integration
make test-e2e
make test

# Tests also run during: make build-builder
```

## Model Examples

The repository includes advanced example scripts for multiple model families:

- `examples/train_and_serve_logistic_regression.py`
- `examples/train_and_serve_random_forest.py`
- `examples/train_and_serve_neural_network.py`

Each script performs a local demonstration flow:

1. Build a model in Python (example-only dependency scope).
2. Export that model to ONNX.
3. Start the real HTTP inference application in a local subprocess.
4. Call `/ready` and `/invocations`.
5. Validate prediction parity and shut the server down.

The core serving package remains inference-only (`model.onnx` + `/invocations`).

## Optional Native Framework Inference

The API can also serve non-ONNX model artifacts on Python as optional paths.

Install optional dependencies:

```bash
uv sync --extra sklearn
uv sync --extra torch
uv sync --extra tensorflow
```

Run with sklearn model artifact:

```bash
export SM_MODEL_DIR=/path/to/model-dir
export MODEL_TYPE=sklearn
export MODEL_FILENAME=model.joblib
uv run --extra sklearn python -m uvicorn serving:application --host 0.0.0.0 --port 8080
```

Run with PyTorch model artifact:

```bash
export SM_MODEL_DIR=/path/to/model-dir
export MODEL_TYPE=pytorch
export MODEL_FILENAME=model.pt
uv run --extra torch python -m uvicorn serving:application --host 0.0.0.0 --port 8080
```

Run with TensorFlow model artifact:

```bash
export SM_MODEL_DIR=/path/to/model-dir
export MODEL_TYPE=tensorflow
export MODEL_FILENAME=model.keras
uv run --extra tensorflow python -m uvicorn serving:application --host 0.0.0.0 --port 8080
```

Notes:

- `MODEL_TYPE=onnx` remains the default production path.
- `MODEL_TYPE=sklearn` requires `scikit-learn` (and `joblib`).
- `MODEL_TYPE=pytorch` requires `torch`.
- `MODEL_TYPE=tensorflow` requires `tensorflow`.

Run them locally with `uv`:

```bash
uv run --extra examples python -m examples.train_and_serve_logistic_regression

uv run --extra examples python -m examples.train_and_serve_random_forest

uv run --extra examples python -m examples.train_and_serve_neural_network
```

### Example Containers

Dedicated example Dockerfiles are available in `container/examples/`:

- `Dockerfile.example-base`
- `Dockerfile.example-logistic-regression`
- `Dockerfile.example-random-forest`
- `Dockerfile.example-neural-network`

Build and run:

```bash
docker build -f container/examples/Dockerfile.example-base -t example-base .
docker build -f container/examples/Dockerfile.example-neural-network -t example-neural-network .
docker run --rm example-neural-network
```

### CI Validation

Workflow `.github/workflows/examples.yml` validates example scripts and containers on push and pull request.

## Tracing and Context Propagation

This service uses OpenTelemetry with OTLP export and stays backend-agnostic:

- Inbound context extraction is handled by FastAPI instrumentation.
- Outbound context propagation is enabled for `requests` and `httpx`.
- Log correlation fields are added by OpenTelemetry logging instrumentation.
- Invocation responses include `x-trace-id` and `x-span-id` headers when a valid span exists.

Configure via environment variables:

- `OTEL_ENABLED=true|false`
- `OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318/v1/traces`
- `OTEL_EXPORTER_OTLP_HEADERS=k1=v1,k2=v2`
- `OTEL_EXPORTER_OTLP_TIMEOUT=10`

The active propagation format remains standards-based (W3C `traceparent`/`tracestate`), so the service can participate in distributed traces across different meshes and backends.
