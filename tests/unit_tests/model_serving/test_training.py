import os
import sys
import types

import pytest

import training
from config import Settings

pytestmark = pytest.mark.unit


class _FakeFrame:
    def __init__(self, columns, rows):
        self.columns = list(columns)
        self.rows = rows
        self.shape = (len(rows), len(columns))

    def __getitem__(self, key):
        idx = self.columns.index(key)
        return [row[idx] for row in self.rows]

    def drop(self, columns):
        keep = [c for c in self.columns if c not in set(columns)]
        idxs = [self.columns.index(c) for c in keep]
        new_rows = [[row[i] for i in idxs] for row in self.rows]
        return _FakeFrame(keep, new_rows)

    def select_dtypes(self, include):
        return self

    def fillna(self, value):
        return self


def _install_fake_sklearn(monkeypatch):
    sklearn = types.ModuleType("sklearn")
    ensemble = types.ModuleType("sklearn.ensemble")
    linear_model = types.ModuleType("sklearn.linear_model")
    pipeline = types.ModuleType("sklearn.pipeline")
    preprocessing = types.ModuleType("sklearn.preprocessing")

    class FakeRF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True

    class FakeLogReg:
        def __init__(self, max_iter):
            self.max_iter = max_iter

    class FakeScaler:
        def __init__(self, with_mean, with_std):
            self.with_mean = with_mean
            self.with_std = with_std

    class FakePipeline:
        def __init__(self, steps):
            self.steps = steps
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True

    ensemble.RandomForestClassifier = FakeRF
    linear_model.LogisticRegression = FakeLogReg
    preprocessing.StandardScaler = FakeScaler
    pipeline.Pipeline = FakePipeline

    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", linear_model)
    monkeypatch.setitem(sys.modules, "sklearn.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", preprocessing)


class TestTrainHelpers:
    def test_find_train_file_prefers_explicit(self, monkeypatch):
        monkeypatch.setenv("TRAIN_FILE", "/tmp/explicit.csv")
        settings = Settings()
        assert training.find_training_data_file(settings) == "/tmp/explicit.csv"

    def test_find_train_file_prefers_parquet_then_csv(self, monkeypatch):
        monkeypatch.setenv("SM_CHANNEL_TRAIN", "/tmp/train")
        settings = Settings()

        calls = {"n": 0}

        def fake_glob(pattern):
            calls["n"] += 1
            if pattern.endswith("*.parquet"):
                return ["/tmp/train/data.parquet"]
            return ["/tmp/train/data.csv"]

        monkeypatch.setattr(training.glob, "glob", fake_glob)
        assert training.find_training_data_file(settings).endswith(".parquet")

    def test_find_train_file_raises_when_missing(self, monkeypatch):
        monkeypatch.setenv("SM_CHANNEL_TRAIN", "/tmp/train")
        settings = Settings()
        monkeypatch.setattr(training.glob, "glob", lambda pattern: [])
        with pytest.raises(FileNotFoundError, match="No training file found"):
            training.find_training_data_file(settings)

    def test_load_table_uses_pandas_reader(self, monkeypatch):
        pd = types.SimpleNamespace(
            read_parquet=lambda p: {"kind": "parquet", "path": p},
            read_csv=lambda p: {"kind": "csv", "path": p},
        )
        monkeypatch.setitem(sys.modules, "pandas", pd)
        assert training.load_training_table("x.parquet")["kind"] == "parquet"
        assert training.load_training_table("x.csv")["kind"] == "csv"

    def test_fit_sklearn_raises_when_target_missing(self, monkeypatch):
        _install_fake_sklearn(monkeypatch)
        settings = Settings()
        df = _FakeFrame(["a", "b"], [[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="TRAIN_TARGET"):
            training.fit_sklearn_model(settings, df)

    def test_fit_sklearn_rf_branch(self, monkeypatch):
        _install_fake_sklearn(monkeypatch)
        monkeypatch.setenv("TRAIN_ALGORITHM", "rf")
        monkeypatch.setenv("TRAIN_TARGET", "target")
        settings = Settings()
        df = _FakeFrame(["x1", "x2", "target"], [[1, 2, 0], [3, 4, 1]])
        model, n_features = training.fit_sklearn_model(settings, df)
        assert n_features == 2
        assert getattr(model, "fitted", False) is True

    def test_fit_sklearn_logreg_branch(self, monkeypatch):
        _install_fake_sklearn(monkeypatch)
        monkeypatch.setenv("TRAIN_ALGORITHM", "logreg")
        monkeypatch.setenv("TRAIN_TARGET", "target")
        settings = Settings()
        df = _FakeFrame(["x1", "x2", "target"], [[1, 2, 0], [3, 4, 1]])
        model, n_features = training.fit_sklearn_model(settings, df)
        assert n_features == 2
        assert getattr(model, "fitted", False) is True

    def test_fit_sklearn_feature_mismatch(self, monkeypatch):
        _install_fake_sklearn(monkeypatch)
        monkeypatch.setenv("TRAIN_TARGET", "target")
        monkeypatch.setenv("TABULAR_NUM_FEATURES", "3")
        settings = Settings()
        df = _FakeFrame(["x1", "x2", "target"], [[1, 2, 0], [3, 4, 1]])
        with pytest.raises(ValueError, match="Feature count mismatch"):
            training.fit_sklearn_model(settings, df)

    def test_save_sklearn_writes_joblib(self, monkeypatch, tmp_path):
        dumped = {}
        joblib = types.SimpleNamespace(
            dump=lambda model, path: dumped.update(path=path, model=model)
        )
        monkeypatch.setitem(sys.modules, "joblib", joblib)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        model = object()
        training.save_sklearn_model(settings, model)
        assert dumped["model"] is model
        assert dumped["path"] == os.path.join(str(tmp_path), "model.joblib")

    def test_export_onnx_writes_file(self, monkeypatch, tmp_path):
        class _FakeOnx:
            def SerializeToString(self):
                return b"onnx-bytes"

        skl2onnx = types.ModuleType("skl2onnx")
        data_types = types.ModuleType("skl2onnx.common.data_types")
        data_types.FloatTensorType = lambda shape: ("float", shape)
        skl2onnx.convert_sklearn = lambda model, initial_types: _FakeOnx()
        monkeypatch.setitem(sys.modules, "skl2onnx", skl2onnx)
        monkeypatch.setitem(sys.modules, "skl2onnx.common.data_types", data_types)

        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        settings = Settings()
        training.export_model_to_onnx(settings, model=object(), n_features=2)
        assert (tmp_path / "model.onnx").read_bytes() == b"onnx-bytes"


class TestTrainMain:
    def test_main_rejects_unsupported_model_type(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "torch")
        with pytest.raises(ValueError, match="supports MODEL_TYPE=sklearn"):
            training.main()

    def test_main_runs_happy_path_without_export(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        monkeypatch.setenv("EXPORT_ONNX", "false")

        calls = []
        monkeypatch.setattr(
            training,
            "find_training_data_file",
            lambda settings: str(tmp_path / "train.csv"),
        )
        monkeypatch.setattr(training, "load_training_table", lambda path: "df")
        monkeypatch.setattr(
            training, "fit_sklearn_model", lambda settings, df: ("model", 2)
        )
        monkeypatch.setattr(
            training,
            "save_sklearn_model",
            lambda settings, model: calls.append(("save", model)),
        )
        monkeypatch.setattr(
            training,
            "export_model_to_onnx",
            lambda settings, model, n: calls.append(("export", n)),
        )
        training.main()
        assert ("save", "model") in calls
        assert not any(c[0] == "export" for c in calls)

    def test_main_runs_with_export(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        monkeypatch.setenv("EXPORT_ONNX", "true")

        calls = []
        monkeypatch.setattr(
            training,
            "find_training_data_file",
            lambda settings: str(tmp_path / "train.csv"),
        )
        monkeypatch.setattr(training, "load_training_table", lambda path: "df")
        monkeypatch.setattr(
            training, "fit_sklearn_model", lambda settings, df: ("model", 5)
        )
        monkeypatch.setattr(
            training,
            "save_sklearn_model",
            lambda settings, model: calls.append(("save", model)),
        )
        monkeypatch.setattr(
            training,
            "export_model_to_onnx",
            lambda settings, model, n_features: calls.append(("export", n_features)),
        )
        training.main()
        assert ("save", "model") in calls
        assert ("export", 5) in calls
