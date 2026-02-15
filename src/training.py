"""Training workflow helpers and CLI entrypoint."""

from __future__ import annotations

import glob
import importlib
import logging
import os
from collections.abc import Collection
from typing import Protocol, cast

from config import Settings

log = logging.getLogger("train")


class SeriesLike(Protocol):
    """Protocol for vector-like target series."""


class DataFrameLike(Protocol):
    """Protocol for pandas-like dataframes used in training."""

    columns: Collection[object]
    shape: tuple[int, int]

    def __getitem__(self: DataFrameLike, item: str) -> SeriesLike:
        """Return one column."""

    def drop(self: DataFrameLike, *, columns: list[str]) -> DataFrameLike:
        """Drop columns."""

    def select_dtypes(self: DataFrameLike, *, include: list[str]) -> DataFrameLike:
        """Select columns by dtype."""

    def fillna(self: DataFrameLike, value: float) -> DataFrameLike:
        """Fill missing values."""


class TrainingDataRepository:
    """Repository responsible for training data discovery and loading."""

    def __init__(self: TrainingDataRepository, settings: Settings) -> None:
        """Initialize repository.

        Parameters
        ----------
        settings : Settings
            Runtime settings.
        """
        self.settings = settings

    def find_training_data_file(self: TrainingDataRepository) -> str:
        """Find input training file from explicit setting or channel directory."""
        if self.settings.train_file:
            return self.settings.train_file
        parquet_candidates = glob.glob(
            os.path.join(self.settings.train_channel, "*.parquet")
        )
        if parquet_candidates:
            return parquet_candidates[0]
        csv_candidates = glob.glob(os.path.join(self.settings.train_channel, "*.csv"))
        if csv_candidates:
            return csv_candidates[0]
        raise FileNotFoundError(
            "No training file found under "
            f"{self.settings.train_channel} (expected .csv or .parquet)"
        )

    @staticmethod
    def load_training_table(path: str) -> DataFrameLike:
        """Load a training table from CSV or Parquet path."""
        pandas_module = importlib.import_module("pandas")

        if path.endswith(".parquet"):
            return cast(DataFrameLike, pandas_module.read_parquet(path))
        return cast(DataFrameLike, pandas_module.read_csv(path))


class SklearnTrainingService:
    """Service that trains, saves, and exports scikit-learn models."""

    def __init__(self: SklearnTrainingService, settings: Settings) -> None:
        """Initialize service with runtime settings."""
        self.settings = settings

    def fit_model(
        self: SklearnTrainingService, dataframe: DataFrameLike
    ) -> tuple[object, int]:
        """Fit a scikit-learn model and return model with feature count."""
        ensemble_module = importlib.import_module("sklearn.ensemble")
        linear_model_module = importlib.import_module("sklearn.linear_model")
        pipeline_module = importlib.import_module("sklearn.pipeline")
        preprocessing_module = importlib.import_module("sklearn.preprocessing")
        random_forest_classifier = ensemble_module.RandomForestClassifier
        logistic_regression = linear_model_module.LogisticRegression
        pipeline_class = pipeline_module.Pipeline
        standard_scaler = preprocessing_module.StandardScaler

        target, features = self._extract_target_and_features(dataframe)

        if self.settings.train_algorithm == "rf":
            model = random_forest_classifier(
                n_estimators=int(os.getenv("RF_N_ESTIMATORS", "300")),
                max_depth=int(os.getenv("RF_MAX_DEPTH", "0")) or None,
                n_jobs=-1,
                random_state=42,
            )
            model.fit(features, target)
            return model, features.shape[1]

        pipeline = pipeline_class(
            [
                ("scaler", standard_scaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    logistic_regression(
                        max_iter=int(os.getenv("LOGREG_MAX_ITER", "200"))
                    ),
                ),
            ]
        )
        pipeline.fit(features, target)
        return pipeline, features.shape[1]

    def save_model(self: SklearnTrainingService, model: object) -> None:
        """Persist fitted scikit-learn model to model directory."""
        joblib_module = importlib.import_module("joblib")

        os.makedirs(self.settings.model_dir, exist_ok=True)
        model_path = os.path.join(self.settings.model_dir, "model.joblib")
        joblib_module.dump(model, model_path)
        log.info("Saved sklearn model: %s", model_path)

    def export_model_to_onnx(
        self: SklearnTrainingService, model: object, n_features: int
    ) -> None:
        """Export fitted model to ONNX format."""
        skl2onnx_module = importlib.import_module("skl2onnx")
        data_types_module = importlib.import_module("skl2onnx.common.data_types")
        convert_sklearn = skl2onnx_module.convert_sklearn
        float_tensor_type = data_types_module.FloatTensorType

        initial_type = [("input", float_tensor_type([None, n_features]))]
        onx = convert_sklearn(model, initial_types=initial_type)

        model_path = os.path.join(self.settings.model_dir, "model.onnx")
        with open(model_path, "wb") as file_handle:
            file_handle.write(onx.SerializeToString())
        log.info("Exported ONNX model: %s", model_path)

    def _extract_target_and_features(
        self: SklearnTrainingService, dataframe: DataFrameLike
    ) -> tuple[SeriesLike, DataFrameLike]:
        """Extract target vector and numeric feature matrix from dataframe."""
        if self.settings.train_target not in dataframe.columns:
            raise ValueError(
                f"TRAIN_TARGET='{self.settings.train_target}' not found in "
                f"columns={list(dataframe.columns)}"
            )

        target = dataframe[self.settings.train_target]
        features = dataframe.drop(columns=[self.settings.train_target])
        features = features.select_dtypes(include=["number"]).fillna(0.0)
        if (
            self.settings.tabular_num_features > 0
            and features.shape[1] != self.settings.tabular_num_features
        ):
            raise ValueError(
                "Feature count mismatch: "
                f"got {features.shape[1]} expected "
                f"TABULAR_NUM_FEATURES={self.settings.tabular_num_features}"
            )
        return target, features


class TrainingWorkflow:
    """Orchestrates the end-to-end training workflow."""

    def __init__(self: TrainingWorkflow, settings: Settings) -> None:
        """Initialize training workflow with settings."""
        self.settings = settings
        self.repository = TrainingDataRepository(settings)
        self.sklearn_training_service = SklearnTrainingService(settings)

    def run(self: TrainingWorkflow) -> None:
        """Execute the full training workflow."""
        self._validate_supported_model_type()
        path = find_training_data_file(self.settings)
        dataframe = load_training_table(path)

        model, n_features = fit_sklearn_model(self.settings, dataframe)
        save_sklearn_model(self.settings, model)

        if self.settings.export_onnx:
            export_model_to_onnx(self.settings, model, n_features)

        log.info("Training complete. Model dir: %s", self.settings.model_dir)

    def _validate_supported_model_type(self: TrainingWorkflow) -> None:
        if self.settings.model_type and self.settings.model_type not in (
            "sklearn",
            "onnx",
        ):
            raise ValueError(
                "Training entrypoint currently supports "
                "MODEL_TYPE=sklearn (and optional ONNX export)"
            )


def find_training_data_file(settings: Settings) -> str:
    """Find training data file for current settings."""
    return TrainingDataRepository(settings).find_training_data_file()


def load_training_table(path: str) -> DataFrameLike:
    """Load training table from file path."""
    return TrainingDataRepository.load_training_table(path)


def fit_sklearn_model(
    settings: Settings, dataframe: DataFrameLike
) -> tuple[object, int]:
    """Fit a scikit-learn model and return model with feature count."""
    return SklearnTrainingService(settings).fit_model(dataframe)


def save_sklearn_model(settings: Settings, model: object) -> None:
    """Save scikit-learn model artifact."""
    SklearnTrainingService(settings).save_model(model)


def export_model_to_onnx(settings: Settings, model: object, n_features: int) -> None:
    """Export model artifact to ONNX."""
    SklearnTrainingService(settings).export_model_to_onnx(model, n_features)


def main() -> None:
    """Run training workflow from environment-derived settings."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info(
        "Training settings: %s",
        {k: v for k, v in settings.model_dump().items() if "otel" not in k.lower()},
    )

    TrainingWorkflow(settings).run()


if __name__ == "__main__":
    main()
