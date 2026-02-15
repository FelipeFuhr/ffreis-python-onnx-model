"""Input parsing and output formatting utilities."""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt

from config import Settings
from parsed_types import ParsedInput


def _strip_params(content_type: str) -> str:
    """Normalize content type by removing MIME parameters."""
    return (content_type or "").split(";")[0].strip().lower()


def _dtype(name: str) -> npt.DTypeLike:
    """Map dtype aliases to NumPy dtype objects."""
    name = (name or "").strip().lower()
    if name in ("float64", "f64"):
        return np.dtype(np.float64)
    if name in ("float32", "f32", ""):
        return np.dtype(np.float32)
    if name in ("int64", "i64"):
        return np.dtype(np.int64)
    if name in ("int32", "i32"):
        return np.dtype(np.int32)
    if name in ("bool", "boolean"):
        return np.dtype(np.bool_)
    raise ValueError(f"Unsupported dtype: {name}")


def _auto_detect_header(lines: list[str], delim: str) -> bool:
    """Infer whether the first CSV row is a header."""
    if not lines:
        return False
    try:
        for t in lines[0].split(delim):
            float(t.strip())
        return False
    except Exception:
        return True


def _parse_col_selector(selector: str, n_cols: int) -> list[int]:
    """Parse column selector syntax into integer indexes."""
    selector = (selector or "").strip()
    if not selector:
        return list(range(n_cols))
    if ":" in selector:
        a, b = selector.split(":", 1)
        start = int(a) if a else 0
        end = int(b) if b else n_cols
        start = max(0, start)
        end = min(n_cols, end)
        return list(range(start, end))
    indexes: list[int] = []
    for tok in selector.split(","):
        tok = tok.strip()
        if tok:
            indexes.append(int(tok))
    return indexes


def _split_ids_and_features(
    data: np.ndarray, settings: Settings
) -> tuple[np.ndarray, dict[str, object] | None]:
    """Split feature columns and optional identifier columns."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={data.shape}")
    n_cols = data.shape[1]

    if not settings.tabular_id_columns and not settings.tabular_feature_columns:
        return data, None

    if settings.tabular_feature_columns:
        feat_idx = _parse_col_selector(settings.tabular_feature_columns, n_cols)
    else:
        id_idx = (
            set(_parse_col_selector(settings.tabular_id_columns, n_cols))
            if settings.tabular_id_columns
            else set()
        )
        feat_idx = [i for i in range(n_cols) if i not in id_idx]

    feature_data = data[:, feat_idx]
    meta = None

    if settings.tabular_id_columns:
        id_indexes = _parse_col_selector(settings.tabular_id_columns, n_cols)
        meta = {"ids": data[:, id_indexes].tolist(), "id_columns": id_indexes}

    return feature_data, meta


def _parse_csv(payload: bytes, settings: Settings) -> np.ndarray:
    """Parse CSV payload into a 2D NumPy array."""
    text = payload.decode("utf-8").strip()
    if not text:
        raise ValueError("Empty CSV payload")

    lines = text.splitlines()
    if settings.csv_skip_blank_lines:
        lines = [ln for ln in lines if ln.strip()]

    if settings.csv_has_header not in ("auto", "true", "false"):
        raise ValueError("CSV_HAS_HEADER must be auto|true|false")

    drop_header = (settings.csv_has_header == "true") or (
        settings.csv_has_header == "auto"
        and _auto_detect_header(lines, settings.csv_delimiter)
    )
    if drop_header and lines:
        lines = lines[1:]

    rows = [[float(x) for x in ln.split(settings.csv_delimiter)] for ln in lines]
    parsed_array = np.asarray(rows)
    if parsed_array.ndim == 1:
        parsed_array = parsed_array.reshape(1, -1)
    return parsed_array


def _parse_json(payload: bytes, settings: Settings) -> object:
    """Parse JSON payload and extract instances when configured."""
    obj = json.loads(payload.decode("utf-8"))
    key = settings.json_key_instances
    return obj[key] if isinstance(obj, dict) and key in obj else obj


def _parse_jsonl(payload: bytes) -> list[object]:
    """Parse JSON Lines payload into record list."""
    lines = payload.decode("utf-8").splitlines()
    records = []
    for ln in lines:
        if ln.strip():
            records.append(json.loads(ln))
    return records


def _load_json_map(s: str) -> dict[str, str] | None:
    """Load JSON mapping string when provided."""
    s = (s or "").strip()
    if not s:
        return None
    loaded = json.loads(s)
    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object mapping")
    mapping: dict[str, str] = {}
    for key, value in loaded.items():
        mapping[str(key)] = str(value)
    return mapping


def _infer_default_onnx_dtype(arr: np.ndarray) -> npt.DTypeLike:
    """Infer default tensor dtype for ONNX inputs."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.dtype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        return np.dtype(np.int64)
    if np.issubdtype(arr.dtype, np.bool_):
        return np.dtype(np.bool_)
    return arr.dtype


class PayloadParser:
    """Parse HTTP payloads into normalized model input objects."""

    def __init__(self: PayloadParser, settings: Settings) -> None:
        """Initialize parser with runtime settings."""
        self.settings = settings

    def parse(self: PayloadParser, payload: bytes, content_type: str) -> ParsedInput:
        """Parse incoming payload using configured input mode."""
        if self.settings.input_mode != "tabular":
            raise ValueError(
                f"INPUT_MODE={self.settings.input_mode} "
                "not implemented (tabular only for now)"
            )

        normalized_content_type = _strip_params(content_type)
        onnx_input_map = _load_json_map(self.settings.onnx_input_map_json)
        if onnx_input_map and self._is_json_content_type(normalized_content_type):
            return self._parse_onnx_multi_input_payload(
                payload=payload,
                content_type=normalized_content_type,
                onnx_input_map=onnx_input_map,
            )

        return self._parse_tabular_payload(
            payload, normalized_content_type, content_type
        )

    def _parse_onnx_multi_input_payload(
        self: PayloadParser,
        payload: bytes,
        content_type: str,
        onnx_input_map: dict[str, str],
    ) -> ParsedInput:
        """Parse JSON/JSONL into named ONNX input tensors."""
        records = self._extract_records(payload, content_type)
        records = self._validate_record_shape(records)

        dtype_map = _load_json_map(self.settings.onnx_input_dtype_map_json) or {}
        tensors: dict[str, np.ndarray] = {}
        batch_sizes: list[int] = []

        for request_key, onnx_input_name in onnx_input_map.items():
            tensor = self._build_tensor_for_input(
                records=records,
                request_key=request_key,
                onnx_input_name=onnx_input_name,
                dtype_map=dtype_map,
            )
            tensors[onnx_input_name] = tensor

            if self.settings.onnx_dynamic_batch:
                if tensor.ndim == 0:
                    raise ValueError(
                        f"ONNX input '{onnx_input_name}' must be at least "
                        "1D (batched), got scalar"
                    )
                batch_sizes.append(tensor.shape[0])

        self._validate_batch_sizes(batch_sizes)
        return ParsedInput(
            X=None,
            tensors=tensors,
            meta={"records": len(records), "mode": "onnx_multi_input"},
        )

    def _parse_tabular_payload(
        self: PayloadParser,
        payload: bytes,
        normalized_content_type: str,
        raw_content_type: str,
    ) -> ParsedInput:
        """Parse tabular payload and return normalized feature matrix."""
        data = self._load_tabular_data(
            payload, normalized_content_type, raw_content_type
        )
        data = self._cast_tabular_data(data)
        feature_data, metadata = _split_ids_and_features(data, self.settings)
        self._validate_feature_count(feature_data)
        return ParsedInput(X=feature_data, tensors=None, meta=metadata)

    def _extract_records(
        self: PayloadParser, payload: bytes, content_type: str
    ) -> object:
        """Extract records for JSON and JSON Lines content types."""
        if content_type in ("application/json", "application/*+json"):
            return _parse_json(payload, self.settings)
        return _parse_jsonl(payload)

    def _validate_record_shape(
        self: PayloadParser, records: object
    ) -> list[dict[str, object]]:
        """Validate ONNX multi-input records shape and type."""
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records:
            raise ValueError(
                "ONNX multi-input mode expects a JSON object "
                "or a non-empty list of objects"
            )
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(
                "ONNX multi-input mode expects each record to be a JSON object"
            )
        return records

    def _build_tensor_for_input(
        self: PayloadParser,
        records: list[dict[str, object]],
        request_key: str,
        onnx_input_name: str,
        dtype_map: dict[str, str],
    ) -> np.ndarray:
        """Build one tensor for an ONNX input from record values."""
        values = []
        for record in records:
            if request_key not in record:
                raise ValueError(
                    f"Missing key '{request_key}' in one of "
                    "the records for ONNX multi-input"
                )
            values.append(record[request_key])

        try:
            tensor = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"Invalid values for ONNX input '{request_key}'") from exc

        dtype_name = dtype_map.get(request_key) or dtype_map.get(onnx_input_name)
        if dtype_name:
            return tensor.astype(_dtype(dtype_name), copy=False)
        return tensor.astype(_infer_default_onnx_dtype(tensor), copy=False)

    def _validate_batch_sizes(self: PayloadParser, batch_sizes: list[int]) -> None:
        """Validate ONNX batch constraints when enabled."""
        if not self.settings.onnx_dynamic_batch:
            return
        if not batch_sizes or any(size <= 0 for size in batch_sizes):
            raise ValueError("ONNX_DYNAMIC_BATCH enabled but batch dimension invalid")
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"ONNX inputs have mismatched batch sizes: {batch_sizes}")

    def _load_tabular_data(
        self: PayloadParser,
        payload: bytes,
        normalized_content_type: str,
        raw_content_type: str,
    ) -> np.ndarray:
        """Load raw tabular payload as NumPy array."""
        if normalized_content_type in ("text/csv", "application/csv"):
            return _parse_csv(payload, self.settings)

        if normalized_content_type in ("application/json", "application/*+json"):
            obj = _parse_json(payload, self.settings)
            if isinstance(obj, dict) and self.settings.jsonl_features_key in obj:
                obj = [obj[self.settings.jsonl_features_key]]
            return np.asarray(obj)

        if normalized_content_type in (
            "application/jsonlines",
            "application/x-jsonlines",
            "application/jsonl",
            "application/x-ndjson",
        ):
            records = _parse_jsonl(payload)
            rows = []
            for record in records:
                if (
                    isinstance(record, dict)
                    and self.settings.jsonl_features_key in record
                ):
                    rows.append(record[self.settings.jsonl_features_key])
                else:
                    rows.append(record)
            return np.asarray(rows)

        raise ValueError(f"Unsupported Content-Type: {raw_content_type}")

    def _cast_tabular_data(self: PayloadParser, data: np.ndarray) -> np.ndarray:
        """Ensure tabular data has expected dimensionality and dtype."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        target_dtype = _dtype(self.settings.tabular_dtype)
        if data.dtype != target_dtype:
            data = data.astype(target_dtype, copy=False)
        return data

    def _validate_feature_count(self: PayloadParser, feature_data: np.ndarray) -> None:
        """Validate feature count against configured expectation."""
        if self.settings.tabular_num_features <= 0:
            return
        if feature_data.shape[1] != self.settings.tabular_num_features:
            raise ValueError(
                "Feature count mismatch: "
                "got "
                f"{feature_data.shape[1]} expected "
                f"TABULAR_NUM_FEATURES={self.settings.tabular_num_features}"
            )

    @staticmethod
    def _is_json_content_type(content_type: str) -> bool:
        return content_type in (
            "application/json",
            "application/*+json",
            "application/jsonlines",
            "application/x-jsonlines",
            "application/jsonl",
            "application/x-ndjson",
        )


class OutputFormatter:
    """Format prediction outputs according to response preferences."""

    def __init__(self: OutputFormatter, settings: Settings) -> None:
        """Initialize formatter with runtime settings."""
        self.settings = settings

    def format(
        self: OutputFormatter, predictions: object, accept: str
    ) -> tuple[str, str]:
        """Format predictions as JSON or CSV."""
        if isinstance(predictions, dict):
            return json.dumps(predictions), "application/json"

        normalized_accept = (
            (accept or self.settings.default_accept).split(",")[0].strip().lower()
        )
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        if normalized_accept in ("text/csv", "application/csv"):
            return self._format_csv(predictions), "text/csv"
        return self._format_json(predictions), "application/json"

    def _format_csv(self: OutputFormatter, predictions: object) -> str:
        """Format scalar/vector/matrix predictions as CSV text."""
        if (
            isinstance(predictions, list)
            and predictions
            and isinstance(predictions[0], list)
        ):
            return "\n".join(
                self.settings.csv_delimiter.join(map(str, row)) for row in predictions
            )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return "\n".join(map(str, predictions))

    def _format_json(self: OutputFormatter, predictions: object) -> str:
        """Format predictions as JSON payload string."""
        if self.settings.predictions_only:
            return json.dumps(predictions)
        return json.dumps({self.settings.json_output_key: predictions})


def parse_payload(payload: bytes, content_type: str, settings: Settings) -> ParsedInput:
    """Parse request payload into normalized input container."""
    return PayloadParser(settings).parse(payload=payload, content_type=content_type)


def format_output(
    predictions: object, accept: str, settings: Settings
) -> tuple[str, str]:
    """Format prediction output and return body plus content type."""
    return OutputFormatter(settings).format(predictions=predictions, accept=accept)
