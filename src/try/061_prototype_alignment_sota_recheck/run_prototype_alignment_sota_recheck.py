from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "061_prototype_alignment_sota_recheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY060_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "060_embedding_topk_prototype_alignment_quickcheck" / "run_embedding_topk_prototype_alignment_quickcheck.py"
TRY026_SUMMARY_PATH = REPO_ROOT / "outputs" / "try" / "026_tinytcn_priority1_quickcheck" / "full19_multiscale_late_fusion_2s_8s_summary.csv"
TRY026_CASE_PATH = REPO_ROOT / "outputs" / "try" / "026_tinytcn_priority1_quickcheck" / "full19_multiscale_late_fusion_2s_8s_case_level.csv"
TRY042_STABILITY_PATH = REPO_ROOT / "outputs" / "try" / "042_rpm_learned_midband_multiseed_stability_check" / "stability_overview.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复核 060 当前版与 final / added SOTA 的差距。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_cache_dir = model_dir / "encoder_checkpoints"
    encoder_cache_dir.mkdir(parents=True, exist_ok=True)

    try060 = load_module("try060_sota_recheck", TRY060_SCRIPT_PATH)
    try058 = load_module("try058_from_061", try060.TRY058_SCRIPT_PATH)
    try053 = try058.load_try053_module()

    final_records = [record for record in try053.scan_dataset_records() if record.is_labeled]
    added_records = try053.load_added_records()
    all_records = sorted([*final_records, *added_records], key=lambda record: record.case_id)
    common_signal_columns = try053.get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: try053.load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    final_case_rows: list[dict[str, object]] = []
    added_case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for holdout in final_records:
        train_records = [record for record in final_records if record.case_id != holdout.case_id]
        fold_case_rows, fold_neighbor_rows, fold_feature_rows = run_fold(
            try060=try060,
            try058=try058,
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            domain_name="final_loco",
            model_dir=model_dir / "final_loco",
            encoder_cache_dir=encoder_cache_dir / "final_loco",
            top_k=args.top_k,
            random_seed=args.random_seed,
            force_retrain=args.force_retrain,
            include_rpm_knn4=True,
        )
        final_case_rows.extend(fold_case_rows)
        neighbor_rows.extend(fold_neighbor_rows)
        feature_rows.extend(fold_feature_rows)

    for holdout in added_records:
        fold_case_rows, fold_neighbor_rows, fold_feature_rows = run_fold(
            try060=try060,
            try058=try058,
            try053=try053,
            train_records=final_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            domain_name="added_external",
            model_dir=model_dir / "added_external",
            encoder_cache_dir=encoder_cache_dir / "added_external",
            top_k=args.top_k,
            random_seed=args.random_seed,
            force_retrain=args.force_retrain,
            include_rpm_knn4=True,
        )
        added_case_rows.extend(fold_case_rows)
        neighbor_rows.extend(fold_neighbor_rows)
        feature_rows.extend(fold_feature_rows)

    final_case_df = pd.DataFrame(final_case_rows)
    added_case_df = pd.DataFrame(added_case_rows)
    neighbor_df = pd.DataFrame(neighbor_rows)
    feature_df = pd.DataFrame(feature_rows)

    final_summary_df = build_summary(final_case_df)
    added_summary_df = build_summary(added_case_df)
    comparison_df = build_sota_comparison(final_summary_df, added_summary_df)

    final_case_df.to_csv(output_dir / "final_case_level_predictions.csv", index=False, encoding="utf-8-sig")
    final_summary_df.to_csv(output_dir / "final_summary.csv", index=False, encoding="utf-8-sig")
    added_case_df.to_csv(output_dir / "added_case_level_predictions.csv", index=False, encoding="utf-8-sig")
    added_summary_df.to_csv(output_dir / "added_summary.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(output_dir / "comparison_to_sota.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(output_dir / "prototype_neighbors.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_dir / "alignment_feature_table.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", final_summary_df, added_summary_df, comparison_df)

    best_final = final_summary_df.iloc[0]
    best_added = added_summary_df.iloc[0]
    print("061 prototype alignment SOTA recheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best final_loco: {best_final['variant_name']} | case_mae={best_final['case_mae']:.4f}")
    print(f"best added_external: {best_added['variant_name']} | case_mae={best_added['case_mae']:.4f}")


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_fold(
    *,
    try060,
    try058,
    try053,
    train_records,
    holdout,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    domain_name: str,
    model_dir: Path,
    encoder_cache_dir: Path,
    top_k: int,
    random_seed: int,
    force_retrain: bool,
    include_rpm_knn4: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_cache_dir.mkdir(parents=True, exist_ok=True)

    case_rows, neighbor_rows, feature_rows = try060.run_holdout_fold(
        try058=try058,
        try053=try053,
        train_records=train_records,
        holdout=holdout,
        cleaned_signal_frames=cleaned_signal_frames,
        domain_name=domain_name,
        model_dir=model_dir,
        encoder_cache_dir=encoder_cache_dir,
        top_k=top_k,
        random_seed=random_seed,
        force_retrain=force_retrain,
    )

    if include_rpm_knn4:
        base_pred = float(try053.predict_rpm_knn4_with_neighbors(train_records, float(holdout.rpm))[0])
        case_rows.insert(
            0,
            {
                "domain": domain_name,
                "variant_name": "rpm_knn4",
                "case_id": holdout.case_id,
                "file_name": holdout.file_name,
                "true_wind_speed": float(holdout.wind_speed),
                "rpm": float(holdout.rpm),
                "pred_wind_speed": base_pred,
                "signed_error": float(base_pred - float(holdout.wind_speed)),
                "abs_error": abs(base_pred - float(holdout.wind_speed)),
            },
        )

    for row in neighbor_rows:
        row["eval_domain"] = domain_name
    for row in feature_rows:
        row["eval_domain"] = domain_name
    return case_rows, neighbor_rows, feature_rows


def build_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_df.groupby("variant_name", sort=False):
        rows.append(
            {
                "variant_name": variant_name,
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_sota_comparison(final_summary_df: pd.DataFrame, added_summary_df: pd.DataFrame) -> pd.DataFrame:
    final_sota_summary = pd.read_csv(TRY026_SUMMARY_PATH, encoding="utf-8-sig")
    final_sota_value = float(
        final_sota_summary.loc[
            final_sota_summary["variant_name"] == "TinyTCN_multiscale_late_fusion_2s_8s",
            "case_mae",
        ].iloc[0]
    )
    added_sota_summary = pd.read_csv(TRY042_STABILITY_PATH, encoding="utf-8-sig")
    added_sota_value = float(
        added_sota_summary.loc[
            added_sota_summary["variant_name"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
            "case_mae_mean",
        ].iloc[0]
    )

    rows: list[dict[str, object]] = []
    for _, row in final_summary_df.iterrows():
        rows.append(
            {
                "target_domain": "final_loco",
                "reference_variant": "TinyTCN_multiscale_late_fusion_2s_8s",
                "reference_case_mae": final_sota_value,
                "current_variant": row["variant_name"],
                "current_case_mae": float(row["case_mae"]),
                "gap_vs_reference": float(row["case_mae"] - final_sota_value),
            }
        )
    for _, row in added_summary_df.iterrows():
        rows.append(
            {
                "target_domain": "added_external",
                "reference_variant": "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
                "reference_case_mae": added_sota_value,
                "current_variant": row["variant_name"],
                "current_case_mae": float(row["case_mae"]),
                "gap_vs_reference": float(row["case_mae"] - added_sota_value),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_domain", "current_case_mae", "current_variant"]).reset_index(drop=True)


def write_summary_markdown(
    output_path: Path,
    final_summary_df: pd.DataFrame,
    added_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    final_block = comparison_df.loc[comparison_df["target_domain"] == "final_loco"].copy()
    added_block = comparison_df.loc[comparison_df["target_domain"] == "added_external"].copy()
    lines = [
        "# prototype alignment SOTA recheck",
        "",
        "## Final vs SOTA",
        "",
    ]
    for _, row in final_block.iterrows():
        lines.append(
            f"- `{row['current_variant']}`: current=`{row['current_case_mae']:.4f}`, "
            f"reference=`{row['reference_case_mae']:.4f}`, gap=`{row['gap_vs_reference']:+.4f}`"
        )

    lines.extend(["", "## Added vs Upper SOTA", ""])
    for _, row in added_block.iterrows():
        lines.append(
            f"- `{row['current_variant']}`: current=`{row['current_case_mae']:.4f}`, "
            f"reference=`{row['reference_case_mae']:.4f}`, gap=`{row['gap_vs_reference']:+.4f}`"
        )

    lines.extend(["", "## Final Ranking", ""])
    for _, row in final_summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`"
        )

    lines.extend(["", "## Added Ranking", ""])
    for _, row in added_summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
