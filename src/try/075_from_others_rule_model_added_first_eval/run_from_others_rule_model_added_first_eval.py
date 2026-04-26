from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "075_from_others_rule_model_added_first_eval"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
SOURCE_SCRIPT_PATH = REPO_ROOT / "src" / "from_others" / "1" / "predict_2.py"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 from_others/1 规则模型的 added-first 表现。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_model = load_module("from_others_rule_model_075", SOURCE_SCRIPT_PATH)
    case_df = build_case_table()

    all_rows: list[dict[str, object]] = []
    for protocol, protocol_df in (
        ("external_loocv", case_df.copy()),
        ("added_to_added2", case_df.loc[case_df["domain"] == "added2"].copy().reset_index(drop=True)),
    ):
        for _, row in protocol_df.iterrows():
            pred = rule_model.predict(str(row["file_path"]))
            if "error" in pred:
                raise RuntimeError(f"规则模型处理失败: case_id={int(row['case_id'])}, error={pred['error']}")
            pred_wind = float(pred["pred_wind"])
            signed_error = pred_wind - float(row["true_wind_speed"])
            all_rows.append(
                {
                    "protocol": protocol,
                    "variant_name": "from_others_rule_model_v1",
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["domain"]),
                    "true_wind_speed": float(row["true_wind_speed"]),
                    "rpm": float(row["rpm"]),
                    "pred_wind_speed": pred_wind,
                    "signed_error": signed_error,
                    "abs_error": abs(signed_error),
                    "fft_rpm": float(pred["fft_rpm"]),
                    "fft_wind": float(pred["fft_wind"]),
                    "knn_wind": float(pred["knn_wind"]),
                    "rms": float(pred["rms"]),
                    "peak": float(pred["peak"]),
                    "crest": float(pred["crest"]),
                    "fft_freq": float(pred["fft_freq"]),
                    "confidence": str(pred["confidence"]),
                }
            )

    case_level_df = pd.DataFrame(all_rows).sort_values(["protocol", "case_id"]).reset_index(drop=True)
    summary_by_protocol_df = build_summary_by_protocol(case_level_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(case_level_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(
        output_dir / "summary_by_protocol_and_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir / "summary.md", summary_by_protocol_df, summary_by_protocol_and_domain_df, case_level_df)

    best_row = summary_by_protocol_df.iloc[0]
    print("075 from_others rule model added-first eval 已完成。")
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
                "rpm": float(rpm),
            }
        )
    return pd.DataFrame(rows)


def summarize_block(block: pd.DataFrame) -> dict[str, object]:
    true_values = block["true_wind_speed"].to_numpy(dtype=float)
    pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    confidence_counts = block["confidence"].value_counts().to_dict()
    return {
        "case_count": int(len(block)),
        "case_mae": float(mean_absolute_error(true_values, pred_values)),
        "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mean_signed_error": float(np.mean(signed_error)),
        "high_conf_count": int(confidence_counts.get("HIGH", 0)),
        "med_conf_count": int(confidence_counts.get("MED", 0)),
        "low_conf_count": int(confidence_counts.get("LOW", 0)),
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


def write_summary_markdown(
    output_path: Path,
    summary_by_protocol_df: pd.DataFrame,
    summary_by_protocol_and_domain_df: pd.DataFrame,
    case_level_df: pd.DataFrame,
) -> None:
    lines = ["# from_others rule model added-first eval", "", "## Summary By Protocol", ""]
    for protocol, block in summary_by_protocol_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`, "
                f"confidence(H/M/L)=`{int(row['high_conf_count'])}/{int(row['med_conf_count'])}/{int(row['low_conf_count'])}`"
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
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`, "
                f"confidence(H/M/L)=`{int(row['high_conf_count'])}/{int(row['med_conf_count'])}/{int(row['low_conf_count'])}`"
            )
        lines.append("")

    lines.append("## Per Case")
    lines.append("")
    for protocol, block in case_level_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in block.sort_values(["case_id"]).iterrows():
            lines.append(
                f"- `case{int(row['case_id'])}`: pred=`{row['pred_wind_speed']:.4f}`, "
                f"abs_error=`{row['abs_error']:.4f}`, signed_error=`{row['signed_error']:+.4f}`, "
                f"fft_rpm=`{row['fft_rpm']:.2f}`, confidence=`{row['confidence']}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
