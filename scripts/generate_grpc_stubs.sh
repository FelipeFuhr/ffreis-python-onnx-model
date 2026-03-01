#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Keep generator/runtime versions pinned for reproducible stubs across CI/local.
GRPCIO_TOOLS_VERSION="${GRPCIO_TOOLS_VERSION:-1.78.0}"
GRPCIO_VERSION="${GRPCIO_VERSION:-1.78.0}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-6.33.5}"

uv run --no-project \
  --with "grpcio-tools==${GRPCIO_TOOLS_VERSION}" \
  --with "grpcio==${GRPCIO_VERSION}" \
  --with "protobuf==${PROTOBUF_VERSION}" \
  python -m grpc_tools.protoc \
  -I proto \
  --python_out=src \
  --grpc_python_out=src \
  proto/onnx_serving_grpc/inference.proto
