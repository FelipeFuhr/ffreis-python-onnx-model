"""Base adapter contracts and adapter factory."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from config import Settings


class BaseAdapter(ABC):
    """Abstract contract for inference adapters."""

    @abstractmethod
    def is_ready(self: BaseAdapter) -> bool:
        """Return whether the adapter is ready to serve predictions.

        Returns
        -------
        bool
            ``True`` when the adapter is fully initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self: BaseAdapter, parsed_input: object) -> object:
        """Run inference for a parsed input payload.

        Parameters
        ----------
        parsed_input : object
            Parsed model input.

        Returns
        -------
        object
            Model prediction output.
        """
        raise NotImplementedError


def load_adapter(settings: Settings) -> BaseAdapter:
    """Instantiate the appropriate adapter for current settings.

    Parameters
    ----------
    settings : Settings
        Runtime configuration.

    Returns
    -------
    BaseAdapter
        Instantiated inference adapter.
    """
    from onnx_adapter import OnnxAdapter

    model_path = os.path.join(
        settings.model_dir, settings.model_filename or "model.onnx"
    )
    if settings.model_type == "onnx" or os.path.exists(model_path):
        return OnnxAdapter(settings)

    if settings.model_type and settings.model_type != "onnx":
        raise RuntimeError(
            f"MODEL_TYPE={settings.model_type} is not implemented in this package"
        )

    raise RuntimeError("Set MODEL_TYPE=onnx or place model.onnx under SM_MODEL_DIR")
