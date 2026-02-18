"""Smoke-check HTTP and gRPC serving endpoints in docker-compose."""

from __future__ import annotations

import json
import os
import time
from urllib.parse import urlparse

import grpc
import httpx

from onnx_serving_grpc import inference_pb2


def _validate_http_base(api_base: str) -> None:
    """Validate API base URL uses HTTP(S) scheme."""
    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "SERVING_API_BASE must use http or https scheme; "
            f"got {parsed.scheme or '<empty>'}"
        )


def _wait_http_ok(
    client: httpx.Client, path: str, timeout_seconds: float = 40.0
) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = client.get(path)
            if response.status_code == 200:
                return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for HTTP 200 at {path}: {last_error}")


def _assert_http(api_base: str) -> bytes:
    _validate_http_base(api_base)
    with httpx.Client(base_url=api_base, timeout=5.0) as client:
        health = _wait_http_ok(client, "/healthz")
        _wait_http_ok(client, "/readyz")

        payload = b"1,2,3\n4,5,6\n"
        response = client.post(
            "/invocations",
            content=payload,
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.content

    assert health is not None
    parsed = json.loads(body.decode("utf-8"))
    assert parsed == [[6.0], [15.0]], parsed
    return body


def _assert_grpc(grpc_target: str) -> bytes:
    with grpc.insecure_channel(grpc_target) as channel:
        live_rpc = channel.unary_unary(
            "/onnxserving.grpc.InferenceService/Live",
            request_serializer=inference_pb2.LiveRequest.SerializeToString,
            response_deserializer=inference_pb2.StatusReply.FromString,
        )
        ready_rpc = channel.unary_unary(
            "/onnxserving.grpc.InferenceService/Ready",
            request_serializer=inference_pb2.ReadyRequest.SerializeToString,
            response_deserializer=inference_pb2.StatusReply.FromString,
        )
        predict_rpc = channel.unary_unary(
            "/onnxserving.grpc.InferenceService/Predict",
            request_serializer=inference_pb2.PredictRequest.SerializeToString,
            response_deserializer=inference_pb2.PredictReply.FromString,
        )

        live = live_rpc(inference_pb2.LiveRequest(), timeout=5.0)
        ready = ready_rpc(inference_pb2.ReadyRequest(), timeout=5.0)
        assert live.ok is True, live
        assert ready.ok is True, ready

        reply = predict_rpc(
            inference_pb2.PredictRequest(
                payload=b"1,2,3\n4,5,6\n",
                content_type="text/csv",
                accept="application/json",
            ),
            timeout=5.0,
        )
        payload = json.loads(reply.body.decode("utf-8"))
        assert payload == [[6.0], [15.0]], payload
        return reply.body


def main() -> None:
    """Run smoke checks and assert parity between API and gRPC outputs."""
    api_base = os.getenv("SERVING_API_BASE", "http://serving-api:8080")
    grpc_target = os.getenv("SERVING_GRPC_TARGET", "serving-grpc:50052")

    http_body = _assert_http(api_base)
    grpc_body = _assert_grpc(grpc_target)

    assert json.loads(http_body.decode("utf-8")) == json.loads(
        grpc_body.decode("utf-8")
    )
    print("serving API and gRPC smoke checks passed")


if __name__ == "__main__":
    main()
