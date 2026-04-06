from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.current.data_loading import DatasetRecord, QUALITY_COLUMNS, TIME_COLUMN
from src.current.features import WindowConfig

from phase1_feature_groups_lib import build_feature_frame, build_loco_split_map, get_group_feature_columns


@dataclass(frozen=True)
class RawDataset:
    meta_df: pd.DataFrame
    windows: np.ndarray


class RawFlattenRidge:
    def __init__(self) -> None:
        self.pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RawFlattenRidge":
        self.pipeline.fit(_flatten_windows(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(_flatten_windows(X))


class RawFlattenMLP:
    def __init__(self) -> None:
        self.pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RawFlattenMLP":
        self.pipeline.fit(_flatten_windows(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(_flatten_windows(X))


class MiniRocketLikeRidge:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.kernels = self._build_kernels()
        self.pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniRocketLikeRidge":
        self.pipeline.fit(self.transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(self.transform(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        feature_rows: list[np.ndarray] = []
        for sample in X:
            sample_features: list[float] = []
            for channel_signal in sample:
                signal = np.asarray(channel_signal, dtype=float)
                for kernel in self.kernels:
                    conv = np.correlate(signal, kernel, mode="valid")
                    sample_features.append(float(np.max(conv)) if conv.size else 0.0)
                    sample_features.append(float(np.mean(conv > 0)) if conv.size else 0.0)
            feature_rows.append(np.array(sample_features, dtype=float))
        return np.vstack(feature_rows)

    def _build_kernels(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        kernels: list[np.ndarray] = []
        for length in (7, 11, 15):
            for _ in range(8):
                kernel = rng.normal(size=length)
                kernel = kernel - np.mean(kernel)
                norm = float(np.linalg.norm(kernel))
                kernels.append(kernel / norm if norm > 0 else np.ones(length, dtype=float))
        return kernels


def build_raw_window_dataset(
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    config: WindowConfig,
) -> RawDataset:
    meta_rows: list[dict[str, object]] = []
    window_arrays: list[np.ndarray] = []
    for record in records:
        signal_df = cleaned_signal_frames[record.case_id]
        numeric_columns = [
            column for column in signal_df.columns if column not in {TIME_COLUMN, *QUALITY_COLUMNS}
        ]
        for _, segment_df in signal_df.groupby("__segment_id", sort=True):
            total_rows = len(segment_df)
            if total_rows < config.window_size:
                continue
            segment_df = segment_df.reset_index(drop=True)
            for start in range(0, total_rows - config.window_size + 1, config.step_size):
                end = start + config.window_size
                window = segment_df.iloc[start:end]
                meta_rows.append(
                    {
                        "case_id": record.case_id,
                        "file_name": record.file_name,
                        "window_index": len(meta_rows),
                        "start_time": window[TIME_COLUMN].iloc[0],
                        "end_time": window[TIME_COLUMN].iloc[-1],
                        "wind_speed": record.wind_speed,
                        "rpm": record.rpm,
                    }
                )
                window_arrays.append(
                    window[numeric_columns].to_numpy(dtype=float, copy=False).T.copy()
                )
    if not window_arrays:
        raise ValueError("未构建出任何原始窗口样本。")
    return RawDataset(meta_df=pd.DataFrame(meta_rows), windows=np.stack(window_arrays, axis=0))


def evaluate_raw_model_loco(
    dataset: RawDataset,
    model_name: str,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = dataset.windows[labeled_mask]
    split_map = build_loco_split_map(labeled_meta)
    y_all = labeled_meta["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[pd.DataFrame] = []

    for _, (train_idx, valid_idx) in split_map.items():
        estimator = build_raw_model(model_name)
        estimator.fit(labeled_windows[train_idx], y_all[train_idx])
        pred = estimator.predict(labeled_windows[valid_idx])
        valid_df = labeled_meta.iloc[valid_idx][
            ["case_id", "file_name", "window_index", "start_time", "end_time", "wind_speed"]
        ].copy()
        valid_df = valid_df.rename(columns={"wind_speed": "true_wind_speed"})
        valid_df["pred_wind_speed"] = pred
        predictions.append(valid_df)

    return pd.concat(predictions, ignore_index=True)


def evaluate_tabular_reference_loco(
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    config: WindowConfig,
    group_name: str = "G6_TIME_FREQ_CROSS",
) -> pd.DataFrame:
    feature_df = build_feature_frame(records, cleaned_signal_frames, config)
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    split_map = build_loco_split_map(labeled_df)
    feature_columns = get_group_feature_columns(feature_df, group_name)
    X_all = labeled_df[feature_columns].to_numpy(dtype=float, copy=False)
    y_all = labeled_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[pd.DataFrame] = []
    for _, (train_idx, valid_idx) in split_map.items():
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        estimator.fit(X_all[train_idx], y_all[train_idx])
        pred = estimator.predict(X_all[valid_idx])
        valid_df = labeled_df.iloc[valid_idx][
            ["case_id", "file_name", "window_index", "start_time", "end_time", "wind_speed"]
        ].copy()
        valid_df = valid_df.rename(columns={"wind_speed": "true_wind_speed"})
        valid_df["pred_wind_speed"] = pred
        predictions.append(valid_df)
    return pd.concat(predictions, ignore_index=True)


def summarize_predictions(
    prediction_frame: pd.DataFrame,
    model_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    case_df = (
        prediction_frame.groupby(["case_id", "file_name", "true_wind_speed"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "pred_mean"})
    )
    case_df["abs_error"] = (case_df["pred_mean"] - case_df["true_wind_speed"]).abs()
    case_df["model_name"] = model_name
    errors = case_df["pred_mean"] - case_df["true_wind_speed"]
    summary_row = {
        "model_name": model_name,
        "case_mae": float(np.mean(np.abs(errors))),
        "case_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "case_mape": float(
            np.mean(np.abs(errors) / case_df["true_wind_speed"].to_numpy(dtype=float)) * 100
        ),
    }
    return summary_row, case_df


def build_raw_model(model_name: str):
    if model_name == "RawFlattenRidge":
        return RawFlattenRidge()
    if model_name == "RawFlattenMLP":
        return RawFlattenMLP()
    if model_name == "MiniRocketLikeRidge":
        return MiniRocketLikeRidge()
    raise ValueError(f"未知原始时序模型: {model_name}")


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)
