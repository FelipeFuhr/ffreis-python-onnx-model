# Incremental Docker Build with Python

A minimal guide to building Python applications with Docker using multi-stage builds for lightweight container images with reproducible builds and testing.

[![Docker Build](https://github.com/FelipeFuhr/ffreis-python-onnx-model/actions/workflows/docker-build.yml/badge.svg?branch=main)](https://github.com/FelipeFuhr/ffreis-python-onnx-model/actions/workflows/docker-build.yml)

## What is this?

This project demonstrates a **multi-stage Docker build** for Python that:

1. **Stage 1 (Builder)**: Installs dependencies with pip, runs tests, and generates a lock file for reproducibility
2. **Stage 2 (Runtime)**: Copies only the necessary application files to a clean, minimal image

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
2. **Base Builder (`ffreis/base-builder`)**: Adds Python, pip, and build tools to the base
3. **Builder (`ffreis/builder`)**: Installs dependencies, runs tests, generates requirements.lock for reproducibility
4. **Base Runner (`ffreis/base-runner`)**: Minimal runtime base with entrypoint script
5. **Runner (`ffreis/runner`)**: Contains only the application code and Python runtime

**Benefits:**

- **Reproducible builds**: Lock file (`requirements.lock`) generated during build ensures consistent dependencies
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
make build-builder           # Build builder (installs deps, runs tests, creates lock file)
make build-base-runner       # Build minimal runner base
make build-runner            # Build final runner image
make build-images            # Build all images at once
```

### Run targets

```bash
make run-app                 # Run the app in runner container
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
├── app/                    # Python application
│   ├── main.py            # Application entry point
│   ├── lib.py             # Library functions
│   ├── pyproject.toml     # Python project configuration & dependencies
│   ├── requirements.txt   # Placeholder for manual dependencies
│   ├── tests/            # Test suite
│   │   └── test_lib.py   # Unit tests
│   └── Makefile          # App-level build targets
├── container/             # Docker multi-stage build files
│   ├── digests.env       # Base image digest pinning
│   ├── Dockerfile.base
│   ├── Dockerfile.base-builder
│   ├── Dockerfile.builder  # Installs deps, runs tests, creates lock
│   ├── Dockerfile.base-runner
│   └── Dockerfile.runner
├── scripts/              # Helper scripts
│   └── entrypoint.sh    # Container entrypoint
├── Makefile             # Main build orchestration
└── .github/
    └── workflows/
        └── docker-build.yml  # CI workflow

## Development Workflow

1. **Add dependencies**: Update `pyproject.toml` with new dependencies
2. **Run locally**: Test your changes locally with `python3 app/main.py`
3. **Build images**: Run `make build-images` to build and test in containers
4. **Tests run automatically**: Builder stage runs `pytest -v` - build fails if tests fail
5. **Lock file generated**: `requirements.lock` is created for reproducible deployments
6. **Deploy**: Use `ffreis/runner` image in production - it's minimal and secure

## Testing

Tests are written in `app/tests/` and run automatically during the Docker build process. The build will fail if any tests fail, ensuring you never deploy broken code.

```bash
# Run tests locally
cd app && pytest -v

# Tests also run during: make build-builder
```
