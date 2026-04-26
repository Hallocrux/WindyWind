from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "078_repo_fft_rpm_blend_quickcheck"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
FFT_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "043_1_fft_rpm_algorithm_search" / "run_fft_rpm_algorithm_search.py"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
RPM_K = 4
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用仓库 FFT 解析 RPM 做 true/fft/0.5 混合 quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fft_module = load_module("repo_fft_078", FFT_SCRIPT_PATH)
    case_df = build_case_table(fft_module)
    fft_feature_df = attach_repo_fft_features(case_df, fft_module)

    loocv_df = run_external_loocv(fft_feature_df)
    transfer_df = run_added_to_added2(fft_feature_df)
    case_level_df = pd.concat([loocv_df, transfer_df], ignore_index=True)
    summary_by_protocol_df = build_summary_by_protocol(case_level_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(case_level_df)

    fft_feature_df.to_csv(output_dir / "fft_feature_table.csv", index=False, encoding="utf-8-sig")
    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(
        output_dir / "summary_by_protocol_and_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir / "summary.md", summary_by_protocol_df, summary_by_protocol_and_domain_df)

    best_row = summary_by_protocol_df.iloc[0]
    print("078 repo fft rpm blend quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best protocol={best_row['protocol']} | "
        f"variant={best_row['variant_name']} | "
        f"case_mae={best_row['case_mae']:.4f}"
    )


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_case_table(fft_module) -> pd.DataFrame:
    from src.current.data_loading import DatasetRecord

    rows: list[dict[str, object]] = []
    for manifest_path, data_dir, domain in (
        (ADDED_MANIFEST_PATH, ADDED_DIR, "added"),
        (ADDED2_MANIFEST_PATH, ADDED2_DIR, "added2"),
    ):
        manifest_df = pd.read_csv(manifest_path)
        for _, row in manifest_df.iterrows():
            wind_speed = row.get("wind_speed")
            rpm = row.get("rpm")
            if pd.isna(wind_speed) or pd.isna(rpm):
                continue
            case_id = int(row["case_id"])
            rows.append(
                {
                    "record": DatasetRecord(
                        case_id=case_id,
                        display_name=str(row["display_name"]),
                        file_name=f"工况{case_id}.csv",
                        file_path=data_dir / f"工况{case_id}.csv",
                        wind_speed=float(wind_speed),
                        rpm=float(rpm),
                        is_labeled=True,
                        original_file_name=str(row.get("original_file_name", "")),
                        label_source=str(row.get("label_source", "")),
                        notes=str(row.get("notes", "")),
                    ),
                    "case_id": case_id,
                    "file_name": f"工况{case_id}.csv",
                    "domain": domain,
                    "true_wind_speed": float(wind_speed),
                    "true_rpm": float(rpm),
                }
            )
    case_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    all_records = case_df["record"].tolist()
    common_signal_columns = fft_module.get_common_signal_columns(all_records)
    cleaned_frames = {
        record.case_id: fft_module.load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }
    case_df["cleaned_frame"] = case_df["record"].map(lambda record: cleaned_frames[record.case_id])
    return case_df


def attach_repo_fft_features(case_df: pd.DataFrame, fft_module) -> pd.DataFrame:
    spec_whole = {spec.variant_name: spec for spec in fft_module.build_variant_specs()}["fft_peak_1x_whole"]
    spec_window8 = {spec.variant_name: spec for spec in fft_module.build_variant_specs()}["window_peak_1x_conf_8s"]

    rows: list[dict[str, object]] = []
    for _, row in case_df.iterrows():
        record = row["record"]
        frame = row["cleaned_frame"]
        whole_est = fft_module.estimate_record_rpm(record=record, frame=frame, spec=spec_whole)
        window_est = fft_module.estimate_record_rpm(record=record, frame=frame, spec=spec_window8)

        pred_whole = float(whole_est.pred_rpm)
        pred_window = float(window_est.pred_rpm)
        if np.isclose(pred_whole, pred_window, atol=1e-9):
            use_window = False
        else:
            use_window = max(pred_whole, pred_window) < 150.0 or pred_window > pred_whole
        hybrid_rpm = pred_window if use_window else pred_whole
        hybrid_source = "window_peak_1x_conf_8s" if use_window else "fft_peak_1x_whole"

        true_rpm = float(row["true_rpm"])
        rows.append(
            {
                "case_id": int(row["case_id"]),
                "file_name": str(row["file_name"]),
                "domain": str(row["domain"]),
                "true_wind_speed": float(row["true_wind_speed"]),
                "true_rpm": true_rpm,
                "repo_fft_rpm": float(hybrid_rpm),
                "repo_fft_source": hybrid_source,
                "repo_fft_confidence": float(window_est.confidence if use_window else whole_est.confidence),
                "repo_fft_whole_rpm": pred_whole,
                "repo_fft_window8_rpm": pred_window,
                "repo_mix05_rpm": 0.5 * true_rpm + 0.5 * float(hybrid_rpm),
                "repo_delta_rpm": float(hybrid_rpm - true_rpm),
                "repo_abs_delta_rpm": abs(float(hybrid_rpm - true_rpm)),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def run_external_loocv(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout_case_id in case_df["case_id"].tolist():
        train_df = case_df.loc[case_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_row = case_df.loc[case_df["case_id"] == holdout_case_id].iloc[0]
        rows.extend(build_case_rows("external_loocv", train_df, test_row))
    return pd.DataFrame(rows)


def run_added_to_added2(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_df = case_df.loc[case_df["domain"] == "added"].copy().reset_index(drop=True)
    for _, test_row in case_df.loc[case_df["domain"] == "added2"].iterrows():
        rows.extend(build_case_rows("added_to_added2", train_df, test_row))
    return pd.DataFrame(rows)


def build_case_rows(protocol: str, train_df: pd.DataFrame, test_row: pd.Series) -> list[dict[str, object]]:
    pred_map = {
        "rpm_true_knn4": predict_rpm_knn4(train_df, float(test_row["true_rpm"])),
        "rpm_repo_fft_knn4": predict_rpm_knn4(train_df, float(test_row["repo_fft_rpm"])),
        "rpm_repo_mix05_knn4": predict_rpm_knn4(train_df, float(test_row["repo_mix05_rpm"])),
    }
    rows: list[dict[str, object]] = []
    for variant_name, pred_wind in pred_map.items():
        signed_error = float(pred_wind - float(test_row["true_wind_speed"]))
        rows.append(
            {
                "protocol": protocol,
                "variant_name": variant_name,
                "case_id": int(test_row["case_id"]),
                "file_name": str(test_row["file_name"]),
                "domain": str(test_row["domain"]),
                "true_wind_speed": float(test_row["true_wind_speed"]),
                "true_rpm": float(test_row["true_rpm"]),
                "repo_fft_rpm": float(test_row["repo_fft_rpm"]),
                "repo_mix05_rpm": float(test_row["repo_mix05_rpm"]),
                "pred_wind_speed": float(pred_wind),
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "repo_fft_source": str(test_row["repo_fft_source"]),
                "repo_fft_confidence": float(test_row["repo_fft_confidence"]),
                "repo_delta_rpm": float(test_row["repo_delta_rpm"]),
                "repo_abs_delta_rpm": float(test_row["repo_abs_delta_rpm"]),
            }
        )
    return rows


def predict_rpm_knn4(train_df: pd.DataFrame, rpm_value: float) -> float:
    train_rpm = train_df["true_rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["true_wind_speed"].to_numpy(dtype=float, copy=False)
    distances = np.abs(train_rpm - rpm_value)
    order = np.argsort(distances)[: min(RPM_K, len(train_df))]
    weights = 1.0 / np.maximum(distances[order], EPS)
    weights = weights / weights.sum()
    return float(np.dot(weights, train_wind[order]))


def summarize_block(block: pd.DataFrame) -> dict[str, object]:
    true_values = block["true_wind_speed"].to_numpy(dtype=float)
    pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    return {
        "case_count": int(len(block)),
        "case_mae": float(mean_absolute_error(true_values, pred_values)),
        "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mean_signed_error": float(np.mean(signed_error)),
    }


def build_summary_by_protocol(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, variant_name), block in case_level_df.groupby(["protocol", "variant_name"], sort=False):
        row = {"protocol": protocol, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_by_protocol_and_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, domain, variant_name), block in case_level_df.groupby(
        ["protocol", "domain", "variant_name"],
        sort=False,
    ):
        row = {"protocol": protocol, "domain": domain, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["protocol", "domain", "case_mae", "case_rmse", "variant_name"])
        .reset_index(drop=True)
    )


def write_summary_markdown(output_path: Path, summary_by_protocol_df: pd.DataFrame, summary_by_protocol_and_domain_df: pd.DataFrame) -> None:
    lines = ["# repo fft rpm blend quickcheck", "", "## Summary By Protocol", ""]
    for protocol, block in summary_by_protocol_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")
    lines.append("## Summary By Protocol And Domain")
    lines.append("")
    for (protocol, domain), block in summary_by_protocol_and_domain_df.groupby(["protocol", "domain"], sort=False):
        lines.append(f"### {protocol} | {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
