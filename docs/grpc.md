# gRPC Interface

The serving package now exposes gRPC endpoints in parallel to HTTP.

HTTP health endpoints are standardized as:

- `GET /healthz` (alias of `/live`)
- `GET /readyz` (alias of `/ready`)

## Install

```bash
uv sync --extra grpc
```

## Run server

Local-only bind (recommended default):

```bash
uv run --extra grpc onnx-model-serving-grpc --host 127.0.0.1 --port 50052
```

Container bind (required when traffic comes from other containers/pods):

```bash
uv run --extra grpc onnx-model-serving-grpc --host 0.0.0.0 --port 50052
```

## Security note

- Default gRPC port is `50052` (override with `--port`).
- Binding to `0.0.0.0` exposes the service on all interfaces in the container/host network namespace.
- In production, prefer private networks and restrict access using firewall rules, Kubernetes NetworkPolicies, service mesh policies, or security groups.
- Avoid publishing the gRPC port publicly unless required. Expose only to trusted peers/load balancers.

## API surface

- `Live` -> process liveness
- `Ready` -> model readiness
- `Predict` -> payload inference

`Predict` fields:

- `payload` (`bytes`)
- `content_type` (defaults to configured HTTP default)
- `accept` (defaults to configured HTTP default)

Proto contract:

- `proto/onnx_serving_grpc/inference.proto`

Generated modules:

- `src/onnx_serving_grpc/inference_pb2.py`
- `src/onnx_serving_grpc/inference_pb2_grpc.py`

These files are generated on demand and are not committed.

## Sync checks

Regenerate:

```bash
make grpc-generate
```

Verify generated files are in sync with proto:

```bash
make grpc-check
```

Remove generated stubs:

```bash
make grpc-clean
```

Run API + gRPC compose smoke:

```bash
make smoke-api-grpc
```
