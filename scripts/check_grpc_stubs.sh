#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

"${ROOT_DIR}/scripts/generate_grpc_stubs.sh"

if [ ! -s src/onnx_serving_grpc/inference_pb2.py ] || [ ! -s src/onnx_serving_grpc/inference_pb2_grpc.py ]; then
  echo "gRPC stub generation failed: expected generated files are missing."
  exit 1
fi

echo "gRPC stubs generated successfully."
