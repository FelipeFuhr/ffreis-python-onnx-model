# Incremental Docker Build with Python

A minimal guide to building Python applications with Docker using multi-stage builds for lightweight container images.

[![Docker Build](https://github.com/FelipeFuhr/ffreis-python-onnx-model/actions/workflows/docker-build.yml/badge.svg?branch=main)](https://github.com/FelipeFuhr/ffreis-python-onnx-model/actions/workflows/docker-build.yml)

## What is this?

This project demonstrates a **multi-stage Docker build** that:

1. **Stage 1 (Builder)**: Installs Python and compiles your application
2. **Stage 2 (Runtime)**: Copies only the application files to a clean image

## Quick Start

### Build all images

```bash
make build-images
```

### Build only the builder (default)

```bash
make build-builder
```

### Run the app container

```bash
docker run ffreis/runner
```

## How it works

This project uses an **incremental multi-image approach** to optimize build times and image sizes:

**Image Layers:**

1. **Base (`ffreis/base`)**: Lightweight Ubuntu 26.04 base image
2. **Base Builder (`ffreis/base-builder`)**: Adds Python and build tools to the base
3. **Builder (`ffreis/builder`)**: Copies your app source and installs dependencies
4. **Base Runner (`ffreis/base-runner`)**: Alternative minimal runtime base
5. **Runner (`ffreis/runner`)**: Copies only the application files, ready to execute

**Benefits:**

- **Layer caching**: Rebuild only changed layers; base images rarely change
- **Reusable base**: Common `ffreis/base` for multiple projects
- **Minimal final images**: Runtime contains only the application, not source or tools
- **Efficient builds**: Parallel dependency installation

## Available Commands

### Build targets

```bash
make build-base              # Build base Ubuntu image
make build-base-builder      # Build base image with Python
make build-builder           # Build builder with app source (default)
make build-base-runner       # Build minimal runner base
make build-runner            # Build final runner image with application
make build-images            # Build all images at once
```

### Run targets

```bash
make run-app                 # Run the compiled app in runner container
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

- **Incremental builds**: Cache base images separately; only rebuild changed layers
- **Smaller final images**: Runtime excludes all build tools and source code
- **Reusable components**: Share `ffreis/base` and Python builder across projects
- **Faster CI/CD**: Docker layer caching speeds up repeated builds
- **Security**: Minimize attack surface by shipping only necessary files
- **Flexibility**: Choose between base images or swap runners easily

## Project Structure

```
.
├── app/                    # Python application
│   ├── main.py            # Application entry point
│   ├── lib.py             # Library functions
│   ├── requirements.txt   # Python dependencies
│   └── Makefile          # App-level build targets
├── container/             # Docker multi-stage build files
│   ├── digests.env       # Base image digest pinning
│   ├── Dockerfile.base
│   ├── Dockerfile.base-builder
│   ├── Dockerfile.builder
│   ├── Dockerfile.base-runner
│   └── Dockerfile.runner
├── scripts/              # Helper scripts
│   └── entrypoint.sh    # Container entrypoint
├── Makefile             # Main build orchestration
└── .github/
    └── workflows/
        └── docker-build.yml  # CI workflow
```
