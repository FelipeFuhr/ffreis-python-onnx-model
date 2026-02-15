"""Shared parsed data structures."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class ParsedInput(BaseModel):
    """Normalized request input consumed by model adapters.

    Parameters
    ----------
    X : numpy.ndarray | None, default=None
        Tabular features represented as a two-dimensional array.
    tensors : dict[str, numpy.ndarray] | None, default=None
        Named tensors for multi-input models.
    meta : dict[str, Any] | None, default=None
        Auxiliary metadata generated during parsing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: np.ndarray | None = None
    tensors: dict[str, np.ndarray] | None = None
    meta: dict[str, Any] | None = None
