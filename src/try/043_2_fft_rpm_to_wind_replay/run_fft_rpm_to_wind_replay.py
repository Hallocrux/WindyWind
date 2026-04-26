from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import DatasetRecord, scan_dataset_records  # noqa: E402

TRY_NAME = "043_2_fft_rpm_to_wind_replay"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
FFT_CASE_PATH = REPO_ROOT / "outputs" / "try" / "043_1_fft_rpm_algorithm_search" / "case_level_predictions.csv"
TCN_WIND_SUMMARY_PATH = REPO_ROOT / "outputs" / "try" / "043_pred_rpm_deployability_check" / "rpm_to_wind_summary.csv"
TCN_WIND_CASE_PATH = REPO_ROOT / "outputs" / "try" / "043_pred_rpm_deployability_check" / "rpm_to_wind_case_level_predictions.csv"
TRY043_PATH = REPO_ROOT / "src" / "try" / "043_pred_rpm_deployability_check" / "run_pred_rpm_deployability_check.py"


@dataclass(frozen=True)
class ReplayContext:
    final_case_info: pd.DataFrame
    added_case_info: pd.DataFrame
    predict_wind_from_rpm: object
    load_added_records: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用已有 FFT RPM 结果回放 rpm->wind。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = build_context()
    fft_case_df = pd.read_csv(FFT_CASE_PATH, encoding="utf-8-sig")
    fft_case_df = fft_case_df.loc[
        fft_case_df["variant_name"].notna()
        & fft_case_df["pred_rpm"].notna()
    ].copy()

    case_level_df = build_fft_wind_case_level_df(fft_case_df, context)
    summary_df = build_summary(case_level_df)
    compare_df = build_compare_df(summary_df)
    write_summary_markdown(args.output_dir / "summary.md", summary_df, compare_df)

    case_level_df.to_csv(
        args.output_dir / "fft_rpm_to_wind_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        args.output_dir / "fft_rpm_to_wind_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compare_df.to_csv(
        args.output_dir / "fft_vs_tcn_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_added = summary_df.loc[summary_df["domain"] == "added_external"].iloc[0]
    print("043_2 FFT RPM -> wind 结果回放完成。")
    print(f"输出目录: {args.output_dir}")
    print(
        f"best added: {best_added['variant_name']} | "
        f"case_mae={best_added['case_mae']:.4f} | mapping={best_added['mapping_method']}"
    )


def build_context() -> ReplayContext:
    try043 = load_try043_module()
    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records: list[DatasetRecord] = try043.load_added_records()
    final_case_info = pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "true_rpm": float(record.rpm),
            }
            for record in final_records
        ]
    ).sort_values("case_id").reset_index(drop=True)
    added_case_info = pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "true_rpm": float(record.rpm),
            }
            for record in added_records
        ]
    ).sort_values("case_id").reset_index(drop=True)
    return ReplayContext(
        final_case_info=final_case_info,
        added_case_info=added_case_info,
        predict_wind_from_rpm=try043.predict_wind_from_rpm,
        load_added_records=try043.load_added_records,
    )


def load_try043_module():
    spec = importlib.util.spec_from_file_location("try043_module", TRY043_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {TRY043_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["try043_module"] = module
    spec.loader.exec_module(module)
    return module


def build_fft_wind_case_level_df(fft_case_df: pd.DataFrame, context: ReplayContext) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mapping_methods = ["rpm_knn4", "rpm_linear", "ridge_rpm_to_wind"]

    for variant_name, variant_df in fft_case_df.groupby("variant_name", sort=True):
        final_variant_df = variant_df.loc[variant_df["domain"] == "final_direct"].copy()
        added_variant_df = variant_df.loc[variant_df["domain"] == "added_external"].copy()
        final_lookup = final_variant_df.set_index("case_id")
        added_lookup = added_variant_df.set_index("case_id")

        for mapping_method in mapping_methods:
            rows.extend(
                replay_final_loco(
                    context=context,
                    final_lookup=final_lookup,
                    mapping_method=mapping_method,
                    variant_name=variant_name,
                )
            )
            rows.extend(
                replay_added_external(
                    context=context,
                    added_lookup=added_lookup,
                    mapping_method=mapping_method,
                    variant_name=variant_name,
                )
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["domain", "case_mae_order", "variant_name", "case_id"])
        .drop(columns=["case_mae_order"])
        .reset_index(drop=True)
    )


def replay_final_loco(
    *,
    context: ReplayContext,
    final_lookup: pd.DataFrame,
    mapping_method: str,
    variant_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id, test_row in context.final_case_info.set_index("case_id").iterrows():
        if case_id not in final_lookup.index:
            continue
        train_df = context.final_case_info.loc[context.final_case_info["case_id"] != case_id].copy()
        input_rpm = float(final_lookup.loc[case_id, "pred_rpm"])
        pred_wind = context.predict_wind_from_rpm(
            train_rpm=train_df["true_rpm"].to_numpy(dtype=float),
            train_wind=train_df["true_wind_speed"].to_numpy(dtype=float),
            input_rpm=np.array([input_rpm], dtype=float),
            mapping_method=mapping_method,
        )[0]
        signed_error = float(pred_wind - float(test_row["true_wind_speed"]))
        rows.append(
            {
                "domain": "final_loco",
                "protocol": "final_loco",
                "variant_name": f"fft_{variant_name}__to__{mapping_method}",
                "fft_variant_name": variant_name,
                "input_rpm_source": "fft_rpm",
                "window_label": "replay",
                "mapping_method": mapping_method,
                "case_id": int(case_id),
                "file_name": str(test_row["file_name"]),
                "true_wind_speed": float(test_row["true_wind_speed"]),
                "true_rpm": float(test_row["true_rpm"]),
                "input_rpm_value": input_rpm,
                "pred_wind_speed": float(pred_wind),
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "case_mae_order": 0,
            }
        )
    return rows


def replay_added_external(
    *,
    context: ReplayContext,
    added_lookup: pd.DataFrame,
    mapping_method: str,
    variant_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train_rpm = context.final_case_info["true_rpm"].to_numpy(dtype=float)
    train_wind = context.final_case_info["true_wind_speed"].to_numpy(dtype=float)
    for _, test_row in context.added_case_info.iterrows():
        case_id = int(test_row["case_id"])
        if case_id not in added_lookup.index:
            continue
        input_rpm = float(added_lookup.loc[case_id, "pred_rpm"])
        pred_wind = context.predict_wind_from_rpm(
            train_rpm=train_rpm,
            train_wind=train_wind,
            input_rpm=np.array([input_rpm], dtype=float),
            mapping_method=mapping_method,
        )[0]
        signed_error = float(pred_wind - float(test_row["true_wind_speed"]))
        rows.append(
            {
                "domain": "added_external",
                "protocol": "added_external",
                "variant_name": f"fft_{variant_name}__to__{mapping_method}",
                "fft_variant_name": variant_name,
                "input_rpm_source": "fft_rpm",
                "window_label": "replay",
                "mapping_method": mapping_method,
                "case_id": case_id,
                "file_name": str(test_row["file_name"]),
                "true_wind_speed": float(test_row["true_wind_speed"]),
                "true_rpm": float(test_row["true_rpm"]),
                "input_rpm_value": input_rpm,
                "pred_wind_speed": float(pred_wind),
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "case_mae_order": 0,
            }
        )
    return rows


def build_summary(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, variant_name), block in case_level_df.groupby(["domain", "variant_name"], sort=False):
        rows.append(
            {
                "domain": domain,
                "protocol": block["protocol"].iloc[0],
                "variant_name": variant_name,
                "fft_variant_name": block["fft_variant_name"].iloc[0],
                "input_rpm_source": "fft_rpm",
                "window_label": "replay",
                "mapping_method": block["mapping_method"].iloc[0],
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case_count": int(len(block)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["domain", "case_mae", "case_rmse", "variant_name"])
        .reset_index(drop=True)
    )


def build_compare_df(fft_summary_df: pd.DataFrame) -> pd.DataFrame:
    tcn_summary_df = pd.read_csv(TCN_WIND_SUMMARY_PATH, encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    for domain in ["final_loco", "added_external"]:
        fft_best = fft_summary_df.loc[fft_summary_df["domain"] == domain].iloc[0]
        tcn_deployable_best = tcn_summary_df.loc[
            (tcn_summary_df["domain"] == domain)
            & (tcn_summary_df["input_rpm_source"] == "pred_rpm")
        ].sort_values(["case_mae", "case_rmse", "variant_name"]).iloc[0]
        true_rpm_upper = tcn_summary_df.loc[
            (tcn_summary_df["domain"] == domain)
            & (tcn_summary_df["input_rpm_source"] == "true_rpm")
        ].sort_values(["case_mae", "case_rmse", "variant_name"]).iloc[0]
        rows.append(
            {
                "domain": domain,
                "fft_best_variant": fft_best["variant_name"],
                "fft_best_case_mae": float(fft_best["case_mae"]),
                "fft_best_case_rmse": float(fft_best["case_rmse"]),
                "fft_mapping_method": fft_best["mapping_method"],
                "tcn_best_variant": tcn_deployable_best["variant_name"],
                "tcn_best_case_mae": float(tcn_deployable_best["case_mae"]),
                "tcn_best_case_rmse": float(tcn_deployable_best["case_rmse"]),
                "true_rpm_upper_variant": true_rpm_upper["variant_name"],
                "true_rpm_upper_case_mae": float(true_rpm_upper["case_mae"]),
                "true_rpm_upper_case_rmse": float(true_rpm_upper["case_rmse"]),
                "delta_fft_minus_tcn_case_mae": float(fft_best["case_mae"] - tcn_deployable_best["case_mae"]),
                "delta_fft_minus_true_rpm_upper_case_mae": float(fft_best["case_mae"] - true_rpm_upper["case_mae"]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(output_path: Path, summary_df: pd.DataFrame, compare_df: pd.DataFrame) -> None:
    final_best = summary_df.loc[summary_df["domain"] == "final_loco"].iloc[0]
    added_best = summary_df.loc[summary_df["domain"] == "added_external"].iloc[0]
    lines = [
        f"# {TRY_NAME} 结论",
        "",
        f"- `final` 最优 FFT deployable 链：`{final_best['variant_name']}` | case_mae=`{final_best['case_mae']:.4f}`",
        f"- `added` 最优 FFT deployable 链：`{added_best['variant_name']}` | case_mae=`{added_best['case_mae']:.4f}`",
        "",
        "## FFT vs TCN",
        "",
    ]
    for _, row in compare_df.iterrows():
        lines.append(f"### {row['domain']}")
        lines.append("")
        lines.append(
            f"- FFT best: `{row['fft_best_variant']}` | case_mae=`{row['fft_best_case_mae']:.4f}` | mapping=`{row['fft_mapping_method']}`"
        )
        lines.append(
            f"- TCN best: `{row['tcn_best_variant']}` | case_mae=`{row['tcn_best_case_mae']:.4f}`"
        )
        lines.append(
            f"- true_rpm upper: `{row['true_rpm_upper_variant']}` | case_mae=`{row['true_rpm_upper_case_mae']:.4f}`"
        )
        lines.append(
            f"- FFT vs TCN delta: `{row['delta_fft_minus_tcn_case_mae']:+.4f}`"
        )
        lines.append(
            f"- FFT vs true_rpm upper delta: `{row['delta_fft_minus_true_rpm_upper_case_mae']:+.4f}`"
        )
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
