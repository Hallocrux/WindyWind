from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import CleaningConfig, DatasetRecord, get_common_signal_columns, prepare_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig, build_case_feature_frame

TRY_NAME = "047_soft_gate_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
BASE_FINAL_PATH = REPO_ROOT / "outputs" / "try" / "026_tinytcn_priority1_quickcheck" / "full19_multiscale_late_fusion_2s_8s_case_level.csv"
BASE_ADDED_PATH = REPO_ROOT / "outputs" / "try" / "035_added_validation_with_full_final_pool" / "added_case_predictions.csv"
ENHANCED_FINAL_DIR = REPO_ROOT / "outputs" / "try" / "044_final_loco_fft_midband_fusion_check" / "seed_runs"
ENHANCED_ADDED_PATH = REPO_ROOT / "outputs" / "try" / "042_rpm_learned_midband_multiseed_stability_check" / "seed_case_level_predictions.csv"
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 case-level 特征做 soft gate quickcheck。")
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

    gate_feature_df = build_gate_feature_table(all_records, common_signal_columns)
    expert_df = build_expert_prediction_table()
    dataset_df = gate_feature_df.merge(expert_df, on=["case_id", "file_name", "true_wind_speed", "domain"], how="inner")
    dataset_df["pred_gap"] = dataset_df["pred_enhanced"] - dataset_df["pred_base"]
    dataset_df["abs_pred_gap"] = dataset_df["pred_gap"].abs()
    dataset_df["optimal_gate_target"] = compute_optimal_gate_target(
        true_values=dataset_df["true_wind_speed"].to_numpy(dtype=float),
        pred_base=dataset_df["pred_base"].to_numpy(dtype=float),
        pred_enhanced=dataset_df["pred_enhanced"].to_numpy(dtype=float),
    )
    dataset_df["enhanced_better"] = (
        (dataset_df["pred_enhanced"] - dataset_df["true_wind_speed"]).abs()
        < (dataset_df["pred_base"] - dataset_df["true_wind_speed"]).abs()
    ).astype(int)

    prediction_rows: list[dict[str, object]] = []
    for test_case_id in dataset_df["case_id"]:
        train_df = dataset_df.loc[dataset_df["case_id"] != test_case_id].copy()
        test_df = dataset_df.loc[dataset_df["case_id"] == test_case_id].copy()
        feature_columns = get_gate_feature_columns()

        global_weight = search_best_global_weight(train_df)
        global_pred = float(
            (1.0 - global_weight) * float(test_df["pred_base"].iloc[0])
            + global_weight * float(test_df["pred_enhanced"].iloc[0])
        )
        prediction_rows.append(
            {
                "variant_name": "global_weight_cv",
                "case_id": int(test_df["case_id"].iloc[0]),
                "file_name": str(test_df["file_name"].iloc[0]),
                "domain": str(test_df["domain"].iloc[0]),
                "true_wind_speed": float(test_df["true_wind_speed"].iloc[0]),
                "pred_base": float(test_df["pred_base"].iloc[0]),
                "pred_enhanced": float(test_df["pred_enhanced"].iloc[0]),
                "pred_gate": global_weight,
                "optimal_gate_target": float(test_df["optimal_gate_target"].iloc[0]),
                "pred_wind_speed": global_pred,
            }
        )

        ridge_gate = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
            ]
        )
        tree_gate = HistGradientBoostingRegressor(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=3,
            l2_regularization=0.1,
            random_state=42,
        )
        for gate_name, estimator in (
            ("ridge_gate", ridge_gate),
            ("hgb_gate", tree_gate),
        ):
            estimator.fit(
                train_df[feature_columns].to_numpy(dtype=float),
                train_df["optimal_gate_target"].to_numpy(dtype=float),
            )
            pred_gate = float(estimator.predict(test_df[feature_columns].to_numpy(dtype=float))[0])
            pred_gate = float(np.clip(pred_gate, 0.0, 1.0))
            pred_wind = float(
                (1.0 - pred_gate) * float(test_df["pred_base"].iloc[0])
                + pred_gate * float(test_df["pred_enhanced"].iloc[0])
            )
            prediction_rows.append(
                {
                    "variant_name": gate_name,
                    "case_id": int(test_df["case_id"].iloc[0]),
                    "file_name": str(test_df["file_name"].iloc[0]),
                    "domain": str(test_df["domain"].iloc[0]),
                    "true_wind_speed": float(test_df["true_wind_speed"].iloc[0]),
                    "pred_base": float(test_df["pred_base"].iloc[0]),
                    "pred_enhanced": float(test_df["pred_enhanced"].iloc[0]),
                    "pred_gate": pred_gate,
                    "optimal_gate_target": float(test_df["optimal_gate_target"].iloc[0]),
                    "pred_wind_speed": pred_wind,
                }
            )

    prediction_df = pd.DataFrame(prediction_rows)
    fixed_rows = build_fixed_variant_rows(dataset_df)
    case_level_df = pd.concat([prediction_df, fixed_rows], ignore_index=True)
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()

    summary_by_variant_df = build_summary_by_variant(case_level_df)
    summary_by_domain_df = build_summary_by_domain(summary_by_variant_df)

    gate_feature_df.to_csv(output_dir / "gate_feature_table.csv", index=False, encoding="utf-8-sig")
    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_variant_df.to_csv(output_dir / "summary_by_variant.csv", index=False, encoding="utf-8-sig")
    summary_by_domain_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_by_variant_df, summary_by_domain_df)

    best_all = summary_by_variant_df.loc[summary_by_variant_df["domain"] == "all_labeled"].iloc[0]
    print("047 soft gate quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best all_labeled: {best_all['variant_name']} | "
        f"case_mae={best_all['case_mae']:.4f}"
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


def build_gate_feature_table(
    records: list[DatasetRecord],
    common_signal_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strain_columns = [column for column in common_signal_columns if "应变" in column]
    acc_columns = [column for column in common_signal_columns if "Acc" in column]
    for record in records:
        cleaned_df, cleaning_stats = prepare_clean_signal_frame(
            record=record,
            common_signal_columns=common_signal_columns,
            cleaning_config=CleaningConfig(),
        )
        raw_feature_df = build_case_feature_frame(record, cleaned_df, WINDOW_CONFIG)
        row = {
            "case_id": record.case_id,
            "file_name": record.file_name,
            "true_wind_speed": float(record.wind_speed),
            "domain": "added" if record.case_id >= 21 else "final",
            "true_rpm": float(record.rpm),
            "missing_ratio_in_common_cols": float(
                cleaning_stats.edge_removed_rows
            ),  # placeholder, overwritten below
            "edge_removed_ratio": float(cleaning_stats.edge_removed_rows / max(len(pd.read_csv(record.file_path)), 1)),
        }
        row["missing_ratio_in_common_cols"] = estimate_missing_ratio(record.file_path, common_signal_columns)
        row.update(summarize_gate_features(raw_feature_df, strain_columns, acc_columns))
        timestamp = pd.to_datetime(raw_feature_df["start_time"]).median()
        if pd.isna(timestamp):
            hour = 0.0
        else:
            hour = float(timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0)
        angle = 2.0 * np.pi * hour / 24.0
        row["hour_sin"] = float(np.sin(angle))
        row["hour_cos"] = float(np.cos(angle))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def estimate_missing_ratio(file_path: Path, common_signal_columns: list[str]) -> float:
    frame = pd.read_csv(file_path, usecols=common_signal_columns)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return float(numeric.isna().sum().sum() / max(len(frame) * len(common_signal_columns), 1))


def summarize_gate_features(
    raw_feature_df: pd.DataFrame,
    strain_columns: list[str],
    acc_columns: list[str],
) -> dict[str, float]:
    def per_window_mean(columns: list[str]) -> pd.Series:
        return raw_feature_df[columns].mean(axis=1)

    strain_low = per_window_mean([f"{column}__fft_band_ratio_0_2hz" for column in strain_columns])
    strain_mid = per_window_mean([f"{column}__fft_band_ratio_2_5hz" for column in strain_columns])
    strain_rms = per_window_mean([f"{column}__rms" for column in strain_columns])
    acc_energy = per_window_mean([f"{column}__fft_total_energy" for column in acc_columns])
    acc_peak = per_window_mean([f"{column}__fft_peak_freq" for column in acc_columns])
    acc_rms = per_window_mean([f"{column}__rms" for column in acc_columns])

    strain_low_ratio_median = float(strain_low.median())
    strain_mid_ratio_median = float(strain_mid.median())
    return {
        "strain_low_ratio_median": strain_low_ratio_median,
        "strain_mid_ratio_median": strain_mid_ratio_median,
        "strain_low_over_mid": float(strain_low_ratio_median / max(strain_mid_ratio_median, EPS)),
        "strain_rms_median": float(strain_rms.median()),
        "acc_energy_median": float(acc_energy.median()),
        "acc_peak_freq_median": float(acc_peak.median()),
        "strain_acc_rms_ratio": float(strain_rms.median() / max(acc_rms.median(), EPS)),
    }


def build_expert_prediction_table() -> pd.DataFrame:
    base_final_df = pd.read_csv(BASE_FINAL_PATH, encoding="utf-8-sig")[
        ["case_id", "file_name", "true_wind_speed", "pred_mean_2s_8s"]
    ].rename(columns={"pred_mean_2s_8s": "pred_base"})
    base_final_df["domain"] = "final"

    base_added_df = pd.read_csv(BASE_ADDED_PATH, encoding="utf-8-sig")[
        ["case_id", "file_name", "true_wind_speed", "pred_2s_8s_fusion"]
    ].rename(columns={"pred_2s_8s_fusion": "pred_base"})
    base_added_df["domain"] = "added"

    enhanced_final_blocks: list[pd.DataFrame] = []
    for path in sorted(ENHANCED_FINAL_DIR.glob("seed_*_case_level_predictions.csv")):
        block = pd.read_csv(path, encoding="utf-8-sig")
        block = block.loc[
            block["variant_name"] == "fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco__w0.3",
            ["case_id", "file_name", "true_wind_speed", "pred_wind_speed"],
        ].copy()
        enhanced_final_blocks.append(block)
    enhanced_final_df = pd.concat(enhanced_final_blocks, ignore_index=True)
    enhanced_final_df = (
        enhanced_final_df.groupby(["case_id", "file_name", "true_wind_speed"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "pred_enhanced"})
    )
    enhanced_final_df["domain"] = "final"

    enhanced_added_df = pd.read_csv(ENHANCED_ADDED_PATH, encoding="utf-8-sig")
    enhanced_added_df = enhanced_added_df.loc[
        enhanced_added_df["variant_name"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
        ["case_id", "file_name", "true_wind_speed", "pred_wind_speed"],
    ].copy()
    enhanced_added_df = (
        enhanced_added_df.groupby(["case_id", "file_name", "true_wind_speed"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "pred_enhanced"})
    )
    enhanced_added_df["domain"] = "added"

    base_df = pd.concat([base_final_df, base_added_df], ignore_index=True)
    enhanced_df = pd.concat([enhanced_final_df, enhanced_added_df], ignore_index=True)
    return base_df.merge(
        enhanced_df,
        on=["case_id", "file_name", "true_wind_speed", "domain"],
        how="inner",
    )


def compute_optimal_gate_target(
    *,
    true_values: np.ndarray,
    pred_base: np.ndarray,
    pred_enhanced: np.ndarray,
) -> np.ndarray:
    denom = pred_enhanced - pred_base
    raw = np.zeros_like(true_values, dtype=float)
    valid = np.abs(denom) > EPS
    raw[valid] = (true_values[valid] - pred_base[valid]) / denom[valid]
    return np.clip(raw, 0.0, 1.0)


def get_gate_feature_columns() -> list[str]:
    return [
        "true_rpm",
        "pred_base",
        "pred_enhanced",
        "pred_gap",
        "abs_pred_gap",
        "hour_sin",
        "hour_cos",
        "strain_low_ratio_median",
        "strain_mid_ratio_median",
        "strain_low_over_mid",
        "strain_rms_median",
        "acc_energy_median",
        "acc_peak_freq_median",
        "strain_acc_rms_ratio",
        "missing_ratio_in_common_cols",
        "edge_removed_ratio",
    ]


def build_fixed_variant_rows(dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in dataset_df.iterrows():
        rows.append(
            {
                "variant_name": "base_only",
                "case_id": int(row["case_id"]),
                "file_name": str(row["file_name"]),
                "domain": str(row["domain"]),
                "true_wind_speed": float(row["true_wind_speed"]),
                "pred_base": float(row["pred_base"]),
                "pred_enhanced": float(row["pred_enhanced"]),
                "pred_gate": 0.0,
                "optimal_gate_target": float(row["optimal_gate_target"]),
                "pred_wind_speed": float(row["pred_base"]),
            }
        )
        rows.append(
            {
                "variant_name": "enhanced_only",
                "case_id": int(row["case_id"]),
                "file_name": str(row["file_name"]),
                "domain": str(row["domain"]),
                "true_wind_speed": float(row["true_wind_speed"]),
                "pred_base": float(row["pred_base"]),
                "pred_enhanced": float(row["pred_enhanced"]),
                "pred_gate": 1.0,
                "optimal_gate_target": float(row["optimal_gate_target"]),
                "pred_wind_speed": float(row["pred_enhanced"]),
            }
        )
        oracle_gate = float(row["optimal_gate_target"])
        rows.append(
            {
                "variant_name": "oracle_soft_gate",
                "case_id": int(row["case_id"]),
                "file_name": str(row["file_name"]),
                "domain": str(row["domain"]),
                "true_wind_speed": float(row["true_wind_speed"]),
                "pred_base": float(row["pred_base"]),
                "pred_enhanced": float(row["pred_enhanced"]),
                "pred_gate": oracle_gate,
                "optimal_gate_target": oracle_gate,
                "pred_wind_speed": float((1.0 - oracle_gate) * row["pred_base"] + oracle_gate * row["pred_enhanced"]),
            }
        )
    return pd.DataFrame(rows)


def search_best_global_weight(train_df: pd.DataFrame) -> float:
    best_weight = 0.0
    best_mae = float("inf")
    for weight in np.linspace(0.0, 1.0, 101):
        pred = (1.0 - weight) * train_df["pred_base"].to_numpy(dtype=float) + weight * train_df["pred_enhanced"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(pred - train_df["true_wind_speed"].to_numpy(dtype=float))))
        if mae < best_mae - 1e-12:
            best_mae = mae
            best_weight = float(weight)
    return best_weight


def build_summary_by_variant(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_level_df.groupby("variant_name", sort=False):
        rows.append(build_summary_row("all_labeled", variant_name, block))
        for domain_name, domain_block in block.groupby("domain", sort=False):
            rows.append(build_summary_row(domain_name, variant_name, domain_block))
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_row(domain: str, variant_name: str, block: pd.DataFrame) -> dict[str, object]:
    gate_values = block["pred_gate"].to_numpy(dtype=float)
    return {
        "domain": domain,
        "variant_name": variant_name,
        "case_mae": float(mean_absolute_error(block["true_wind_speed"], block["pred_wind_speed"])),
        "case_rmse": float(np.sqrt(mean_squared_error(block["true_wind_speed"], block["pred_wind_speed"]))),
        "mean_signed_error": float(block["signed_error"].mean()),
        "mean_gate": float(np.mean(gate_values)),
        "gate_std": float(np.std(gate_values)),
        "case_count": int(len(block)),
    }


def build_summary_by_domain(summary_by_variant_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, block in summary_by_variant_df.groupby("domain", sort=False):
        best_row = block.iloc[0]
        rows.append(
            {
                "domain": domain,
                "best_variant": best_row["variant_name"],
                "best_case_mae": float(best_row["case_mae"]),
                "base_only_case_mae": float(block.loc[block["variant_name"] == "base_only", "case_mae"].iloc[0]),
                "enhanced_only_case_mae": float(block.loc[block["variant_name"] == "enhanced_only", "case_mae"].iloc[0]),
                "oracle_soft_gate_case_mae": float(block.loc[block["variant_name"] == "oracle_soft_gate", "case_mae"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(
    output_path: Path,
    summary_by_variant_df: pd.DataFrame,
    summary_by_domain_df: pd.DataFrame,
) -> None:
    lines = [
        f"# {TRY_NAME}",
        "",
        "## Domain Summary",
        "",
    ]
    for _, row in summary_by_domain_df.iterrows():
        lines.append(f"### {row['domain']}")
        lines.append("")
        lines.append(f"- best variant: `{row['best_variant']}` | case_mae=`{row['best_case_mae']:.4f}`")
        lines.append(f"- `base_only`: case_mae=`{row['base_only_case_mae']:.4f}`")
        lines.append(f"- `enhanced_only`: case_mae=`{row['enhanced_only_case_mae']:.4f}`")
        lines.append(f"- `oracle_soft_gate`: case_mae=`{row['oracle_soft_gate_case_mae']:.4f}`")
        lines.append("")

    lines.extend(["## Variant Ranking", ""])
    for domain, block in summary_by_variant_df.groupby("domain", sort=False):
        lines.append(f"### {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_gate=`{row['mean_gate']:.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
