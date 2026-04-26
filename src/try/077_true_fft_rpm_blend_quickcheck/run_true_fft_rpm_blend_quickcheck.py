from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "077_true_fft_rpm_blend_quickcheck"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
FFT_SCRIPT_PATH = REPO_ROOT / "src" / "from_others" / "1" / "predict_2.py"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
RPM_K = 4
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="quickcheck true_rpm 与 fft_rpm 的 0.5 混合是否有正信号。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fft_model = load_module("fft_rule_077", FFT_SCRIPT_PATH)
    case_df = build_case_table()
    fft_feature_df = attach_fft_features(case_df, fft_model)

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
    print("077 true fft rpm blend quickcheck 已完成。")
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


def build_case_table() -> pd.DataFrame:
    frames = [
        load_manifest_as_cases(ADDED_MANIFEST_PATH, ADDED_DIR, "added"),
        load_manifest_as_cases(ADDED2_MANIFEST_PATH, ADDED2_DIR, "added2"),
    ]
    return pd.concat(frames, ignore_index=True).sort_values("case_id").reset_index(drop=True)


def load_manifest_as_cases(manifest_path: Path, data_dir: Path, domain: str) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        wind_speed = row.get("wind_speed")
        rpm = row.get("rpm")
        if pd.isna(wind_speed) or pd.isna(rpm):
            continue
        case_id = int(row["case_id"])
        rows.append(
            {
                "case_id": case_id,
                "file_name": f"工况{case_id}.csv",
                "file_path": data_dir / f"工况{case_id}.csv",
                "domain": domain,
                "true_wind_speed": float(wind_speed),
                "true_rpm": float(rpm),
            }
        )
    return pd.DataFrame(rows)


def attach_fft_features(case_df: pd.DataFrame, fft_model) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in case_df.iterrows():
        pred = fft_model.predict(str(row["file_path"]))
        if "error" in pred:
            raise RuntimeError(f"FFT 提取失败: case_id={int(row['case_id'])}, error={pred['error']}")
        fft_rpm = float(pred["fft_rpm"])
        true_rpm = float(row["true_rpm"])
        rows.append(
            {
                **row.to_dict(),
                "fft_rpm": fft_rpm,
                "rpm_mix05": 0.5 * true_rpm + 0.5 * fft_rpm,
                "fft_confidence": str(pred["confidence"]),
                "fft_wind_rule": float(pred["fft_wind"]),
                "knn_wind_rule": float(pred["knn_wind"]),
                "pred_wind_rule": float(pred["pred_wind"]),
                "delta_rpm": fft_rpm - true_rpm,
                "abs_delta_rpm": abs(fft_rpm - true_rpm),
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
        "rpm_fft_knn4": predict_rpm_knn4(train_df, float(test_row["fft_rpm"])),
        "rpm_mix05_knn4": predict_rpm_knn4(train_df, float(test_row["rpm_mix05"])),
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
                "fft_rpm": float(test_row["fft_rpm"]),
                "rpm_mix05": float(test_row["rpm_mix05"]),
                "pred_wind_speed": float(pred_wind),
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "fft_confidence": str(test_row["fft_confidence"]),
                "delta_rpm": float(test_row["delta_rpm"]),
                "abs_delta_rpm": float(test_row["abs_delta_rpm"]),
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
    lines = ["# true fft rpm blend quickcheck", "", "## Summary By Protocol", ""]
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
