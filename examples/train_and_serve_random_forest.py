"""Train and serve a random forest ONNX model."""

from __future__ import annotations

from examples.common import run_train_and_serve_demo


def main() -> None:
    """Run random forest training and serving demo."""
    run_train_and_serve_demo("random_forest")


if __name__ == "__main__":
    main()
