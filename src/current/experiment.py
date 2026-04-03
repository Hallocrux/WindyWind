from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import WINDOW_META_COLUMNS, get_vibration_feature_columns

OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class ExperimentSpec:
    feature_set: str
    model_name: str
    feature_columns: list[str]


def run_model_comparison(feature_df: pd.DataFrame) -> dict[str, object]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    unlabeled_df = feature_df[feature_df["wind_speed"].isna()].copy()
    _validate_feature_frame(labeled_df)

    vibration_columns = get_vibration_feature_columns(feature_df)
    experiment_specs = _build_experiment_specs(vibration_columns)
    feature_columns_map = _get_feature_columns_map(vibration_columns)

    window_level_frames: list[pd.DataFrame] = []
    case_level_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    labeled_case_ids = sorted(labeled_df["case_id"].unique())

    for spec in experiment_specs:
        prediction_frame = _evaluate_single_spec(labeled_df, labeled_case_ids, spec)
        window_level_frames.append(prediction_frame)

        case_frame = (
            prediction_frame.groupby(
                ["model_name", "feature_set", "case_id", "file_name", "true_wind_speed"],
                as_index=False,
            )["pred_wind_speed"]
            .mean()
            .rename(columns={"pred_wind_speed": "pred_mean"})
        )
        case_frame["abs_error"] = (
            case_frame["pred_mean"] - case_frame["true_wind_speed"]
        ).abs()
        case_level_frames.append(case_frame)
        summary_rows.append(_summarize_case_predictions(case_frame))

    window_level = pd.concat(window_level_frames, ignore_index=True)
    case_level = pd.concat(case_level_frames, ignore_index=True)
    model_summary = pd.DataFrame(summary_rows).sort_values(
        ["case_mae", "case_rmse", "model_name", "feature_set"]
    )
    model_summary["feature_columns"] = model_summary.apply(
        lambda row: feature_columns_map[(row["feature_set"], row["model_name"])],
        axis=1,
    )

    best_model = select_best_model(model_summary)
    unlabeled_selection = select_unlabeled_model(model_summary, unlabeled_df)
    unlabeled_predictions = predict_unlabeled_cases(
        labeled_df=labeled_df,
        unlabeled_df=unlabeled_df,
        spec=unlabeled_selection["spec"],
        selection_reason=unlabeled_selection["reason"],
        best_overall=best_model,
    )

    return {
        "window_level_predictions": window_level,
        "case_level_predictions": case_level,
        "model_summary": model_summary.drop(columns=["feature_columns"]).reset_index(drop=True),
        "best_model": best_model,
        "unlabeled_predictions": unlabeled_predictions,
        "labeled_case_count": len(labeled_case_ids),
        "total_window_count": len(feature_df),
    }


def save_outputs(results: dict[str, object], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results["model_summary"]).to_csv(
        output_dir / "model_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(results["case_level_predictions"]).to_csv(
        output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(results["window_level_predictions"]).to_csv(
        output_dir / "window_level_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(results["unlabeled_predictions"]).to_csv(
        output_dir / "unlabeled_predictions.csv", index=False, encoding="utf-8-sig"
    )


def format_console_summary(results: dict[str, object]) -> str:
    best_model = results["best_model"]
    unlabeled_predictions = pd.DataFrame(results["unlabeled_predictions"])
    unlabeled_line = "无无标签工况预测结果。"
    if not unlabeled_predictions.empty:
        first_row = unlabeled_predictions.iloc[0]
        unlabeled_line = (
            f"{first_row['file_name']} 预测风速: {first_row['predicted_wind_speed']:.4f} m/s"
        )

    return "\n".join(
        [
            f"带标签工况数: {results['labeled_case_count']}",
            f"总窗口数: {results['total_window_count']}",
            f"最优模型: {best_model['model_name']}",
            f"最优特征集: {best_model['feature_set']}",
            f"最优 case_mae: {best_model['case_mae']:.4f}",
            unlabeled_line,
        ]
    )


def select_best_model(model_summary: pd.DataFrame) -> dict[str, object]:
    min_mae = float(model_summary["case_mae"].min())
    mae_candidates = model_summary[model_summary["case_mae"] <= min_mae + 0.02].copy()
    min_rmse = float(mae_candidates["case_rmse"].min())
    rmse_candidates = mae_candidates[mae_candidates["case_rmse"] == min_rmse].copy()
    rmse_candidates["simplicity_rank"] = rmse_candidates["model_name"].map(
        {
            "LinearRegression": 0,
            "Ridge": 1,
            "HistGradientBoostingRegressor": 2,
            "RandomForestRegressor": 3,
        }
    )
    best_row = rmse_candidates.sort_values(
        ["simplicity_rank", "model_name", "feature_set"]
    ).iloc[0]
    return best_row.drop(labels=["simplicity_rank"]).to_dict()


def select_unlabeled_model(
    model_summary: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
) -> dict[str, object]:
    if unlabeled_df.empty:
        spec_row = select_best_model(model_summary)
        return {
            "spec": _row_to_spec(spec_row),
            "reason": "无无标签工况，沿用全局最优模型。",
        }

    has_rpm = unlabeled_df["rpm"].notna().all()
    compatible = model_summary.copy()
    if not has_rpm:
        compatible = compatible[compatible["feature_set"] == "VIB_FT"].copy()

    if compatible.empty:
        raise ValueError("无可用于无标签工况推理的兼容模型。")

    best_compatible = select_best_model(compatible)
    reason = "使用全局最优模型进行无标签推理。"
    best_overall = select_best_model(model_summary)
    if (
        best_compatible["model_name"] != best_overall["model_name"]
        or best_compatible["feature_set"] != best_overall["feature_set"]
    ):
        reason = (
            "全局最优模型需要 rpm 特征，但无标签工况缺少 rpm，"
            "因此回退到最佳 rpm-free 模型。"
        )

    return {"spec": _row_to_spec(best_compatible), "reason": reason}


def predict_unlabeled_cases(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    spec: ExperimentSpec,
    selection_reason: str,
    best_overall: dict[str, object],
) -> pd.DataFrame:
    if unlabeled_df.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "file_name",
                "predicted_wind_speed",
                "model_name",
                "feature_set",
                "selection_reason",
                "best_overall_model_name",
                "best_overall_feature_set",
            ]
        )

    estimator = _build_estimator(spec.model_name)
    estimator.fit(labeled_df[spec.feature_columns], labeled_df["wind_speed"])
    predictions = estimator.predict(unlabeled_df[spec.feature_columns])

    prediction_df = unlabeled_df[["case_id", "file_name"]].copy()
    prediction_df["pred_wind_speed"] = predictions
    case_predictions = (
        prediction_df.groupby(["case_id", "file_name"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "predicted_wind_speed"})
    )
    case_predictions["model_name"] = spec.model_name
    case_predictions["feature_set"] = spec.feature_set
    case_predictions["selection_reason"] = selection_reason
    case_predictions["best_overall_model_name"] = best_overall["model_name"]
    case_predictions["best_overall_feature_set"] = best_overall["feature_set"]
    return case_predictions


def _evaluate_single_spec(
    labeled_df: pd.DataFrame,
    labeled_case_ids: list[int],
    spec: ExperimentSpec,
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for case_id in labeled_case_ids:
        train_df = labeled_df[labeled_df["case_id"] != case_id]
        valid_df = labeled_df[labeled_df["case_id"] == case_id]
        estimator = _build_estimator(spec.model_name)
        estimator.fit(train_df[spec.feature_columns], train_df["wind_speed"])
        pred = estimator.predict(valid_df[spec.feature_columns])

        fold_df = valid_df[
            ["case_id", "file_name", "window_index", "start_time", "end_time", "wind_speed"]
        ].copy()
        fold_df = fold_df.rename(columns={"wind_speed": "true_wind_speed"})
        fold_df["pred_wind_speed"] = pred
        fold_df["model_name"] = spec.model_name
        fold_df["feature_set"] = spec.feature_set
        predictions.append(fold_df)

    return pd.concat(predictions, ignore_index=True)


def _summarize_case_predictions(case_frame: pd.DataFrame) -> dict[str, float | str]:
    errors = case_frame["pred_mean"] - case_frame["true_wind_speed"]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mape = float(
        np.mean(np.abs(errors) / case_frame["true_wind_speed"].to_numpy(dtype=float)) * 100
    )
    first_row = case_frame.iloc[0]
    return {
        "model_name": first_row["model_name"],
        "feature_set": first_row["feature_set"],
        "case_mae": mae,
        "case_rmse": rmse,
        "case_mape": mape,
    }


def _build_experiment_specs(vibration_columns: list[str]) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            feature_set="RPM_ONLY",
            model_name="LinearRegression",
            feature_columns=["rpm"],
        ),
        ExperimentSpec(
            feature_set="VIB_FT",
            model_name="Ridge",
            feature_columns=vibration_columns,
        ),
        ExperimentSpec(
            feature_set="VIB_FT",
            model_name="RandomForestRegressor",
            feature_columns=vibration_columns,
        ),
        ExperimentSpec(
            feature_set="VIB_FT",
            model_name="HistGradientBoostingRegressor",
            feature_columns=vibration_columns,
        ),
        ExperimentSpec(
            feature_set="VIB_FT_RPM",
            model_name="Ridge",
            feature_columns=[*vibration_columns, "rpm"],
        ),
        ExperimentSpec(
            feature_set="VIB_FT_RPM",
            model_name="RandomForestRegressor",
            feature_columns=[*vibration_columns, "rpm"],
        ),
        ExperimentSpec(
            feature_set="VIB_FT_RPM",
            model_name="HistGradientBoostingRegressor",
            feature_columns=[*vibration_columns, "rpm"],
        ),
    ]


def _build_estimator(model_name: str):
    if model_name == "LinearRegression":
        return LinearRegression()
    if model_name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        )
    if model_name == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        )
    raise ValueError(f"未知模型: {model_name}")


def _row_to_spec(row: dict[str, object]) -> ExperimentSpec:
    vibration_columns = row.get("feature_columns")
    if vibration_columns is not None:
        return ExperimentSpec(
            feature_set=str(row["feature_set"]),
            model_name=str(row["model_name"]),
            feature_columns=list(vibration_columns),
        )
    raise ValueError("缺少 feature_columns，无法构造实验规格。")


def _validate_feature_frame(labeled_df: pd.DataFrame) -> None:
    if labeled_df.empty:
        raise ValueError("没有可用于训练的带标签窗口。")

    numeric_columns = [
        column for column in labeled_df.columns if column not in {"file_name", "start_time", "end_time"}
    ]
    numeric_block = labeled_df[numeric_columns].select_dtypes(include=["number"])
    values = numeric_block.to_numpy(dtype=float, copy=False)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("特征矩阵包含 NaN 或 Inf。")


def _get_feature_columns_map(vibration_columns: list[str]) -> dict[tuple[str, str], list[str]]:
    return {
        ("RPM_ONLY", "LinearRegression"): ["rpm"],
        ("VIB_FT", "Ridge"): vibration_columns,
        ("VIB_FT", "RandomForestRegressor"): vibration_columns,
        ("VIB_FT", "HistGradientBoostingRegressor"): vibration_columns,
        ("VIB_FT_RPM", "Ridge"): [*vibration_columns, "rpm"],
        ("VIB_FT_RPM", "RandomForestRegressor"): [*vibration_columns, "rpm"],
        ("VIB_FT_RPM", "HistGradientBoostingRegressor"): [*vibration_columns, "rpm"],
    }
