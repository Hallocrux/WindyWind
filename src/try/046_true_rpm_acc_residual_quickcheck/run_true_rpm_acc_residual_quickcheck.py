from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig, build_case_feature_frame

TRY_NAME = "046_true_rpm_acc_residual_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RESIDUAL_WEIGHTS = (1.0, 0.4)


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    rpm_model_name: str
    uses_acc_residual: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速验证 true rpm 主干与 acc residual 修正。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)
    acc_columns = [column for column in common_signal_columns if "Acc" in column]

    cleaned_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }
    window_feature_frames = build_window_feature_frames(all_records, cleaned_frames, acc_columns)
    feature_table_df = pd.concat(window_feature_frames.values(), ignore_index=True)

    case_level_rows: list[dict[str, object]] = []
    final_rows = evaluate_final_loco(
        final_records=final_records,
        window_feature_frames=window_feature_frames,
    )
    case_level_rows.extend(final_rows)

    added_rows = evaluate_added_external(
        final_records=final_records,
        added_records=added_records,
        window_feature_frames=window_feature_frames,
    )
    case_level_rows.extend(added_rows)

    case_level_df = pd.DataFrame(case_level_rows)
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()

    summary_by_variant_df = build_summary_by_variant(case_level_df)
    summary_by_protocol_df = build_summary_by_protocol(summary_by_variant_df)
    write_summary_markdown(output_dir / "summary.md", summary_by_variant_df, summary_by_protocol_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_variant_df.to_csv(output_dir / "summary_by_variant.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    feature_table_df.to_csv(output_dir / "feature_table.csv", index=False, encoding="utf-8-sig")

    best_final = summary_by_variant_df.loc[summary_by_variant_df["protocol"] == "final_loco"].iloc[0]
    best_added = summary_by_variant_df.loc[summary_by_variant_df["protocol"] == "added_external"].iloc[0]
    print("046 true rpm + acc residual quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        "best final: "
        f"{best_final['variant_name']} | case_mae={best_final['case_mae']:.4f}"
    )
    print(
        "best added: "
        f"{best_added['variant_name']} | case_mae={best_added['case_mae']:.4f}"
    )


def load_added_records() -> list[DatasetRecord]:
    manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=ADDED_STANDARD_DIR / f"工况{case_id}.csv",
                wind_speed=float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None,
                rpm=float(row["rpm"]) if not pd.isna(row["rpm"]) else None,
                is_labeled=not pd.isna(row["wind_speed"]) and not pd.isna(row["rpm"]),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return records


def build_window_feature_frames(
    records: list[DatasetRecord],
    cleaned_frames: dict[int, pd.DataFrame],
    acc_columns: list[str],
) -> dict[int, pd.DataFrame]:
    result: dict[int, pd.DataFrame] = {}
    for record in records:
        raw_feature_df = build_case_feature_frame(record, cleaned_frames[record.case_id], WINDOW_CONFIG)
        result[record.case_id] = compress_acc_feature_frame(raw_feature_df, acc_columns)
    return result


def compress_acc_feature_frame(
    raw_feature_df: pd.DataFrame,
    acc_columns: list[str],
) -> pd.DataFrame:
    feature_df = raw_feature_df[
        [
            "case_id",
            "file_name",
            "window_index",
            "start_time",
            "end_time",
            "wind_speed",
            "rpm",
            "raw_missing_ratio",
            "raw_missing_rows",
            "touches_leading_missing",
            "touches_trailing_missing",
        ]
    ].copy()

    feature_df["acc_rms_mean"] = raw_feature_df[[f"{column}__rms" for column in acc_columns]].mean(axis=1)
    feature_df["acc_std_mean"] = raw_feature_df[[f"{column}__std" for column in acc_columns]].mean(axis=1)
    feature_df["acc_ptp_mean"] = raw_feature_df[[f"{column}__ptp" for column in acc_columns]].mean(axis=1)
    feature_df["acc_peak_freq_mean"] = raw_feature_df[[f"{column}__fft_peak_freq" for column in acc_columns]].mean(axis=1)
    feature_df["acc_band_ratio_0_2hz_mean"] = raw_feature_df[
        [f"{column}__fft_band_ratio_0_2hz" for column in acc_columns]
    ].mean(axis=1)
    feature_df["acc_band_ratio_2_5hz_mean"] = raw_feature_df[
        [f"{column}__fft_band_ratio_2_5hz" for column in acc_columns]
    ].mean(axis=1)
    feature_df["acc_band_ratio_5_10hz_mean"] = raw_feature_df[
        [f"{column}__fft_band_ratio_5_10hz" for column in acc_columns]
    ].mean(axis=1)
    feature_df["acc_log_energy_mean"] = np.log1p(
        raw_feature_df[[f"{column}__fft_total_energy" for column in acc_columns]].mean(axis=1)
    )
    feature_df["acc_mid_over_low_ratio"] = feature_df["acc_band_ratio_2_5hz_mean"] / np.maximum(
        feature_df["acc_band_ratio_0_2hz_mean"],
        1e-6,
    )
    feature_df["acc_high_over_mid_ratio"] = feature_df["acc_band_ratio_5_10hz_mean"] / np.maximum(
        feature_df["acc_band_ratio_2_5hz_mean"],
        1e-6,
    )
    return feature_df


def evaluate_final_loco(
    *,
    final_records: list[DatasetRecord],
    window_feature_frames: dict[int, pd.DataFrame],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test_record in final_records:
        train_records = [record for record in final_records if record.case_id != test_record.case_id]
        for rpm_model_name, rpm_model in build_rpm_models().items():
            train_case_df = build_case_table(train_records)
            train_case_df["base_pred_oof"] = compute_case_oof_predictions(train_case_df, rpm_model_name)
            test_base_pred = predict_case_from_training(
                train_case_df=train_case_df,
                rpm_value=float(test_record.rpm),
                rpm_model_name=rpm_model_name,
            )

            test_case_row = build_case_row(test_record)
            rows.append(
                build_case_prediction_row(
                    protocol="final_loco",
                    domain="final",
                    variant_name=f"rpm_only__{rpm_model_name}",
                    rpm_model_name=rpm_model_name,
                    residual_model_name="none",
                    record=test_record,
                    pred_wind_speed=test_base_pred,
                )
            )

            residual_model = fit_residual_model(
                train_case_df=train_case_df,
                window_feature_frames=window_feature_frames,
            )
            residual_clip_abs = float(train_case_df["residual_oof"].abs().quantile(0.95))
            for residual_weight in RESIDUAL_WEIGHTS:
                test_pred = predict_with_residual(
                    base_pred=test_base_pred,
                    record=test_record,
                    residual_model=residual_model,
                    feature_df=window_feature_frames[test_record.case_id],
                    residual_clip_abs=residual_clip_abs,
                    residual_weight=residual_weight,
                )
                residual_label = format_residual_label(residual_weight)
                rows.append(
                    build_case_prediction_row(
                        protocol="final_loco",
                        domain="final",
                        variant_name=f"rpm_{rpm_model_name}__plus__{residual_label}",
                        rpm_model_name=rpm_model_name,
                        residual_model_name=residual_label,
                        record=test_record,
                        pred_wind_speed=test_pred,
                    )
                )
    return rows


def evaluate_added_external(
    *,
    final_records: list[DatasetRecord],
    added_records: list[DatasetRecord],
    window_feature_frames: dict[int, pd.DataFrame],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train_case_df = build_case_table(final_records)
    for rpm_model_name in build_rpm_models():
        train_case_df_variant = train_case_df.copy()
        train_case_df_variant["base_pred_oof"] = compute_case_oof_predictions(train_case_df_variant, rpm_model_name)
        residual_model = fit_residual_model(
            train_case_df=train_case_df_variant,
            window_feature_frames=window_feature_frames,
        )
        residual_clip_abs = float(train_case_df_variant["residual_oof"].abs().quantile(0.95))
        for record in added_records:
            base_pred = predict_case_from_training(
                train_case_df=train_case_df_variant,
                rpm_value=float(record.rpm),
                rpm_model_name=rpm_model_name,
            )
            rows.append(
                build_case_prediction_row(
                    protocol="added_external",
                    domain="added",
                    variant_name=f"rpm_only__{rpm_model_name}",
                    rpm_model_name=rpm_model_name,
                    residual_model_name="none",
                    record=record,
                    pred_wind_speed=base_pred,
                )
            )
            for residual_weight in RESIDUAL_WEIGHTS:
                corrected_pred = predict_with_residual(
                    base_pred=base_pred,
                    record=record,
                    residual_model=residual_model,
                    feature_df=window_feature_frames[record.case_id],
                    residual_clip_abs=residual_clip_abs,
                    residual_weight=residual_weight,
                )
                residual_label = format_residual_label(residual_weight)
                rows.append(
                    build_case_prediction_row(
                        protocol="added_external",
                        domain="added",
                        variant_name=f"rpm_{rpm_model_name}__plus__{residual_label}",
                        rpm_model_name=rpm_model_name,
                        residual_model_name=residual_label,
                        record=record,
                        pred_wind_speed=corrected_pred,
                    )
                )
    return rows


def build_case_table(records: list[DatasetRecord]) -> pd.DataFrame:
    return pd.DataFrame([build_case_row(record) for record in records]).sort_values("case_id").reset_index(drop=True)


def build_case_row(record: DatasetRecord) -> dict[str, object]:
    return {
        "case_id": record.case_id,
        "file_name": record.file_name,
        "wind_speed": float(record.wind_speed),
        "rpm": float(record.rpm),
    }


def build_rpm_models() -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=RIDGE_ALPHAS),
        "knn4": KNeighborsRegressor(n_neighbors=4, weights="distance"),
        "spline_gam": Pipeline(
            steps=[
                ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
            ]
        ),
    }


def compute_case_oof_predictions(train_case_df: pd.DataFrame, rpm_model_name: str) -> np.ndarray:
    preds: list[float] = []
    for case_id in train_case_df["case_id"]:
        inner_train_df = train_case_df.loc[train_case_df["case_id"] != case_id].copy()
        rpm_value = float(train_case_df.loc[train_case_df["case_id"] == case_id, "rpm"].iloc[0])
        pred = predict_case_from_training(
            train_case_df=inner_train_df,
            rpm_value=rpm_value,
            rpm_model_name=rpm_model_name,
        )
        preds.append(pred)
    return np.asarray(preds, dtype=float)


def predict_case_from_training(
    *,
    train_case_df: pd.DataFrame,
    rpm_value: float,
    rpm_model_name: str,
) -> float:
    estimator = clone(build_rpm_models()[rpm_model_name])
    X_train = train_case_df[["rpm"]].to_numpy(dtype=float)
    y_train = train_case_df["wind_speed"].to_numpy(dtype=float)
    estimator.fit(X_train, y_train)
    pred = estimator.predict(np.asarray([[rpm_value]], dtype=float))[0]
    return float(pred)


def fit_residual_model(
    *,
    train_case_df: pd.DataFrame,
    window_feature_frames: dict[int, pd.DataFrame],
) -> Pipeline:
    feature_rows: list[pd.DataFrame] = []
    residual_lookup = train_case_df.set_index("case_id")["base_pred_oof"]
    for case_id in train_case_df["case_id"]:
        block = window_feature_frames[int(case_id)].copy()
        base_pred = float(residual_lookup.loc[int(case_id)])
        block["base_pred"] = base_pred
        block["residual_target"] = float(block["wind_speed"].iloc[0]) - base_pred
        feature_rows.append(block)
    train_window_df = pd.concat(feature_rows, ignore_index=True)
    train_case_df["residual_oof"] = train_case_df["wind_speed"] - train_case_df["base_pred_oof"]

    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    estimator.fit(
        train_window_df[get_residual_feature_columns()].to_numpy(dtype=float),
        train_window_df["residual_target"].to_numpy(dtype=float),
    )
    return estimator


def predict_with_residual(
    *,
    base_pred: float,
    record: DatasetRecord,
    residual_model: Pipeline,
    feature_df: pd.DataFrame,
    residual_clip_abs: float,
    residual_weight: float,
) -> float:
    eval_df = feature_df.copy()
    eval_df["base_pred"] = base_pred
    residual_window_pred = residual_model.predict(eval_df[get_residual_feature_columns()].to_numpy(dtype=float))
    residual_case_pred = float(np.mean(residual_window_pred))
    clip_value = max(residual_clip_abs, 0.15)
    residual_case_pred = float(np.clip(residual_case_pred, -clip_value, clip_value))
    return float(base_pred + residual_weight * residual_case_pred)


def format_residual_label(residual_weight: float) -> str:
    if abs(residual_weight - 1.0) < 1e-9:
        return "acc_residual_ridge"
    return f"acc_residual_ridge_w{residual_weight:.1f}"


def get_residual_feature_columns() -> list[str]:
    return [
        "base_pred",
        "rpm",
        "acc_rms_mean",
        "acc_std_mean",
        "acc_ptp_mean",
        "acc_peak_freq_mean",
        "acc_log_energy_mean",
        "acc_band_ratio_0_2hz_mean",
        "acc_band_ratio_2_5hz_mean",
        "acc_band_ratio_5_10hz_mean",
        "acc_mid_over_low_ratio",
        "acc_high_over_mid_ratio",
        "raw_missing_ratio",
    ]


def build_case_prediction_row(
    *,
    protocol: str,
    domain: str,
    variant_name: str,
    rpm_model_name: str,
    residual_model_name: str,
    record: DatasetRecord,
    pred_wind_speed: float,
) -> dict[str, object]:
    return {
        "protocol": protocol,
        "domain": domain,
        "variant_name": variant_name,
        "rpm_model_name": rpm_model_name,
        "residual_model_name": residual_model_name,
        "case_id": record.case_id,
        "file_name": record.file_name,
        "true_wind_speed": float(record.wind_speed),
        "rpm": float(record.rpm),
        "pred_wind_speed": float(pred_wind_speed),
    }


def build_summary_by_variant(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, variant_name), block in case_level_df.groupby(["protocol", "variant_name"], sort=False):
        rows.append(
            {
                "protocol": protocol,
                "domain": block["domain"].iloc[0],
                "variant_name": variant_name,
                "rpm_model_name": block["rpm_model_name"].iloc[0],
                "residual_model_name": block["residual_model_name"].iloc[0],
                "case_mae": float(mean_absolute_error(block["true_wind_speed"], block["pred_wind_speed"])),
                "case_rmse": float(np.sqrt(mean_squared_error(block["true_wind_speed"], block["pred_wind_speed"]))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_by_protocol(summary_by_variant_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, block in summary_by_variant_df.groupby("protocol", sort=False):
        best_rpm_only = block.loc[block["residual_model_name"] == "none"].sort_values(["case_mae", "case_rmse"]).iloc[0]
        best_residual = block.loc[block["residual_model_name"] != "none"].sort_values(["case_mae", "case_rmse"]).iloc[0]
        rows.append(
            {
                "protocol": protocol,
                "best_rpm_only_variant": best_rpm_only["variant_name"],
                "best_rpm_only_case_mae": float(best_rpm_only["case_mae"]),
                "best_residual_variant": best_residual["variant_name"],
                "best_residual_case_mae": float(best_residual["case_mae"]),
                "delta_residual_minus_rpm_only": float(best_residual["case_mae"] - best_rpm_only["case_mae"]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(
    output_path: Path,
    summary_by_variant_df: pd.DataFrame,
    summary_by_protocol_df: pd.DataFrame,
) -> None:
    lines = [
        f"# {TRY_NAME}",
        "",
        "## 协议结论",
        "",
    ]
    for _, row in summary_by_protocol_df.iterrows():
        lines.append(f"### {row['protocol']}")
        lines.append("")
        lines.append(
            f"- best rpm-only: `{row['best_rpm_only_variant']}` | case_mae=`{row['best_rpm_only_case_mae']:.4f}`"
        )
        lines.append(
            f"- best residual: `{row['best_residual_variant']}` | case_mae=`{row['best_residual_case_mae']:.4f}`"
        )
        lines.append(
            f"- residual delta vs rpm-only: `{row['delta_residual_minus_rpm_only']:+.4f}`"
        )
        lines.append("")

    lines.extend(["## 变体排名", ""])
    for protocol, block in summary_by_variant_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
