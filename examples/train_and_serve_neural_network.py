"""Train and serve a neural network ONNX model using local HTTP flow."""

from __future__ import annotations

from examples.common import run_train_and_serve_demo


def main() -> None:
    """Run neural-network training and serving demo."""
    run_train_and_serve_demo("neural_network")


if __name__ == "__main__":
    main()
