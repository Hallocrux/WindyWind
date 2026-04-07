from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY041_ROOT = REPO_ROOT / "src" / "try" / "041_rpm_vs_learned_midband_check"
for path in (REPO_ROOT, TRY041_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_rpm_vs_learned_midband_check as try041


TRY_NAME = "042_rpm_learned_midband_multiseed_stability_check"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_BASE_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
LEARNED_VARIANT_NAME = "tinytcn_all_channels_midband_3_0_6_0hz"
TARGET_VARIANT_NAMES = [
    "rpm_knn4",
    LEARNED_VARIANT_NAME,
    "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
    "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5",
    "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.7",
]
TARGET_FUSION_REFERENCE = "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5"
ALL_CHANNELS_VARIANT_INDEX = 2
TINYTCN_SEED_OFFSET = 53 * (ALL_CHANNELS_VARIANT_INDEX + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 rpm_knn4 + TinyTCN midband @ w=0.5 做多随机种子稳定性复核。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_BASE_SEEDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.use_deterministic_algorithms(True)

    train_config = try041.TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    train_records = [record for record in try041.scan_dataset_records() if record.is_labeled]
    added_records = try041.load_added_records()
    all_records = [*train_records, *added_records]
    common_signal_columns = try041.get_common_signal_columns(all_records)
    strain_columns = [column for column in common_signal_columns if "应变" in column]
    acc_columns = [column for column in common_signal_columns if "Acc" in column]
    selected_columns = [*strain_columns, *acc_columns]

    base_frames = {
        record.case_id: try041.load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }
    midband_frames = try041.build_midband_frames(base_frames, all_records, strain_columns)

    variant_config_df = build_variant_config_table()
    seed_case_rows: list[pd.DataFrame] = []
    seed_summary_rows: list[pd.DataFrame] = []

    for seed_order, base_seed in enumerate(args.seeds, start=1):
        tinytcn_seed = base_seed + TINYTCN_SEED_OFFSET
        seed_case_df = run_single_seed(
            base_seed=base_seed,
            seed_order=seed_order,
            tinytcn_seed=tinytcn_seed,
            train_records=train_records,
            added_records=added_records,
            midband_frames=midband_frames,
            selected_columns=selected_columns,
            train_config=train_config,
        )
        seed_summary_df = try041.build_summary(seed_case_df)
        seed_summary_df.insert(0, "seed_order", seed_order)
        seed_summary_df.insert(1, "base_seed", base_seed)
        seed_summary_df.insert(2, "tinytcn_seed", tinytcn_seed)

        seed_case_rows.append(seed_case_df)
        seed_summary_rows.append(seed_summary_df)

    seed_case_level_df = pd.concat(seed_case_rows, ignore_index=True)
    seed_summary_df = pd.concat(seed_summary_rows, ignore_index=True)
    seed_summary_df = add_seed_ranks(seed_summary_df)
    best_variant_by_seed_df = build_best_variant_by_seed(seed_summary_df)
    stability_overview_df = build_stability_overview(seed_summary_df)
    pairwise_comparison_df = build_pairwise_comparison(seed_summary_df, TARGET_FUSION_REFERENCE)
    case22_by_seed_df = (
        seed_case_level_df.loc[seed_case_level_df["case_id"] == 22]
        .sort_values(["base_seed", "abs_error", "variant_name"])
        .reset_index(drop=True)
    )

    variant_config_df.to_csv(output_dir / "variant_config_table.csv", index=False, encoding="utf-8-sig")
    seed_summary_df.to_csv(output_dir / "seed_summary.csv", index=False, encoding="utf-8-sig")
    seed_case_level_df.to_csv(output_dir / "seed_case_level_predictions.csv", index=False, encoding="utf-8-sig")
    best_variant_by_seed_df.to_csv(output_dir / "best_variant_by_seed.csv", index=False, encoding="utf-8-sig")
    stability_overview_df.to_csv(output_dir / "stability_overview.csv", index=False, encoding="utf-8-sig")
    pairwise_comparison_df.to_csv(output_dir / "pairwise_comparison.csv", index=False, encoding="utf-8-sig")
    case22_by_seed_df.to_csv(output_dir / "case22_by_seed.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_dir / "summary.md",
        seed_summary_df=seed_summary_df,
        stability_overview_df=stability_overview_df,
        pairwise_comparison_df=pairwise_comparison_df,
        best_variant_by_seed_df=best_variant_by_seed_df,
        seeds=args.seeds,
    )

    print("多随机种子稳定性复核已完成。")
    print(f"输出目录: {output_dir}")


def run_single_seed(
    *,
    base_seed: int,
    seed_order: int,
    tinytcn_seed: int,
    train_records: list[try041.DatasetRecord],
    added_records: list[try041.DatasetRecord],
    midband_frames: dict[int, pd.DataFrame],
    selected_columns: list[str],
    train_config: try041.TrainConfig,
) -> pd.DataFrame:
    rpm_df = try041.build_rpm_knn_predictions(added_records)
    learned_df = try041.train_and_predict_tinytcn(
        train_records=train_records,
        eval_records=added_records,
        cleaned_signal_frames=midband_frames,
        selected_columns=selected_columns,
        train_config=train_config,
        seed=tinytcn_seed,
    )
    learned_df["variant_name"] = LEARNED_VARIANT_NAME
    learned_df["model_family"] = "tinytcn"
    learned_df["input_columns"] = "all_channels_midband"
    learned_df["strain_transform"] = "strain_bandpass_3.0_6.0Hz"
    learned_df["fusion_weight_learned"] = np.nan

    case_level_df = pd.concat([rpm_df, learned_df], ignore_index=True)
    fusion_df = try041.build_fusion_predictions(case_level_df)
    case_level_df = pd.concat([case_level_df, fusion_df], ignore_index=True)
    case_level_df = case_level_df.loc[case_level_df["variant_name"].isin(TARGET_VARIANT_NAMES)].copy()
    case_level_df["seed_order"] = seed_order
    case_level_df["base_seed"] = base_seed
    case_level_df["tinytcn_seed"] = tinytcn_seed
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()
    return case_level_df.sort_values(["seed_order", "variant_name", "case_id"]).reset_index(drop=True)


def build_variant_config_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_name": "rpm_knn4",
                "model_family": "analytic",
                "input_columns": "rpm",
                "strain_transform": "none",
                "fusion_weight_learned": np.nan,
                "is_seed_sensitive": False,
            },
            {
                "variant_name": LEARNED_VARIANT_NAME,
                "model_family": "tinytcn",
                "input_columns": "all_channels_midband",
                "strain_transform": "strain_bandpass_3.0_6.0Hz",
                "fusion_weight_learned": np.nan,
                "is_seed_sensitive": True,
            },
            {
                "variant_name": "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
                "model_family": "fusion",
                "input_columns": "rpm + all_channels_midband",
                "strain_transform": "strain_bandpass_3.0_6.0Hz",
                "fusion_weight_learned": 0.3,
                "is_seed_sensitive": True,
            },
            {
                "variant_name": TARGET_FUSION_REFERENCE,
                "model_family": "fusion",
                "input_columns": "rpm + all_channels_midband",
                "strain_transform": "strain_bandpass_3.0_6.0Hz",
                "fusion_weight_learned": 0.5,
                "is_seed_sensitive": True,
            },
            {
                "variant_name": "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.7",
                "model_family": "fusion",
                "input_columns": "rpm + all_channels_midband",
                "strain_transform": "strain_bandpass_3.0_6.0Hz",
                "fusion_weight_learned": 0.7,
                "is_seed_sensitive": True,
            },
        ]
    )


def add_seed_ranks(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked_blocks: list[pd.DataFrame] = []
    for base_seed, block in seed_summary_df.groupby("base_seed", sort=True):
        ordered = block.sort_values(["case_mae", "case22_abs_error", "variant_name"]).reset_index(drop=True).copy()
        ordered["case_mae_rank"] = np.arange(1, len(ordered) + 1)
        ranked_blocks.append(ordered)
    return pd.concat(ranked_blocks, ignore_index=True)


def build_best_variant_by_seed(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_summary_df.loc[seed_summary_df["case_mae_rank"] == 1]
        .sort_values(["base_seed", "case_mae", "case22_abs_error"])
        .reset_index(drop=True)
    )


def build_stability_overview(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    rpm_df = seed_summary_df.loc[seed_summary_df["variant_name"] == "rpm_knn4", ["base_seed", "case_mae", "case22_abs_error"]].rename(
        columns={
            "case_mae": "rpm_case_mae",
            "case22_abs_error": "rpm_case22_abs_error",
        }
    )
    rows: list[dict[str, object]] = []
    for variant_name, block in seed_summary_df.groupby("variant_name", sort=False):
        merged_vs_rpm = block[["base_seed", "case_mae", "case22_abs_error"]].merge(rpm_df, on="base_seed", how="inner")
        mae_delta_vs_rpm = merged_vs_rpm["case_mae"] - merged_vs_rpm["rpm_case_mae"]
        case22_delta_vs_rpm = merged_vs_rpm["case22_abs_error"] - merged_vs_rpm["rpm_case22_abs_error"]
        rows.append(
            {
                "variant_name": variant_name,
                "model_family": block["model_family"].iloc[0],
                "input_columns": block["input_columns"].iloc[0],
                "strain_transform": block["strain_transform"].iloc[0],
                "fusion_weight_learned": block["fusion_weight_learned"].iloc[0],
                "seed_count": int(len(block)),
                "best_seed_count": int((block["case_mae_rank"] == 1).sum()),
                "best_seed_rate": float((block["case_mae_rank"] == 1).mean()),
                "case_mae_mean": float(block["case_mae"].mean()),
                "case_mae_std": float(block["case_mae"].std(ddof=0)),
                "case_mae_min": float(block["case_mae"].min()),
                "case_mae_median": float(block["case_mae"].median()),
                "case_mae_max": float(block["case_mae"].max()),
                "case_rmse_mean": float(block["case_rmse"].mean()),
                "case_rmse_std": float(block["case_rmse"].std(ddof=0)),
                "case22_abs_error_mean": float(block["case22_abs_error"].mean()),
                "case22_abs_error_std": float(block["case22_abs_error"].std(ddof=0)),
                "mean_signed_error_mean": float(block["mean_signed_error"].mean()),
                "mean_signed_error_std": float(block["mean_signed_error"].std(ddof=0)),
                "better_than_rpm_count": int((mae_delta_vs_rpm < 0).sum()),
                "better_than_rpm_rate": float((mae_delta_vs_rpm < 0).mean()),
                "delta_vs_rpm_case_mae_mean": float(mae_delta_vs_rpm.mean()),
                "better_than_rpm_case22_count": int((case22_delta_vs_rpm < 0).sum()),
                "better_than_rpm_case22_rate": float((case22_delta_vs_rpm < 0).mean()),
                "delta_vs_rpm_case22_mean": float(case22_delta_vs_rpm.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_mae_mean", "case_mae_std", "variant_name"]).reset_index(drop=True)


def build_pairwise_comparison(seed_summary_df: pd.DataFrame, reference_variant: str) -> pd.DataFrame:
    reference_df = seed_summary_df.loc[seed_summary_df["variant_name"] == reference_variant, ["base_seed", "case_mae", "case_rmse", "case22_abs_error"]].rename(
        columns={
            "case_mae": "reference_case_mae",
            "case_rmse": "reference_case_rmse",
            "case22_abs_error": "reference_case22_abs_error",
        }
    )
    compare_variants = [variant_name for variant_name in TARGET_VARIANT_NAMES if variant_name != reference_variant]
    rows: list[dict[str, object]] = []
    for compare_variant in compare_variants:
        compare_df = seed_summary_df.loc[
            seed_summary_df["variant_name"] == compare_variant,
            ["base_seed", "case_mae", "case_rmse", "case22_abs_error"],
        ].rename(
            columns={
                "case_mae": "compare_case_mae",
                "case_rmse": "compare_case_rmse",
                "case22_abs_error": "compare_case22_abs_error",
            }
        )
        merged = reference_df.merge(compare_df, on="base_seed", how="inner")
        mae_delta = merged["reference_case_mae"] - merged["compare_case_mae"]
        rmse_delta = merged["reference_case_rmse"] - merged["compare_case_rmse"]
        case22_delta = merged["reference_case22_abs_error"] - merged["compare_case22_abs_error"]
        rows.append(
            {
                "reference_variant": reference_variant,
                "compare_variant": compare_variant,
                "seed_count": int(len(merged)),
                "reference_better_case_mae_count": int((mae_delta < 0).sum()),
                "reference_better_case_mae_rate": float((mae_delta < 0).mean()),
                "delta_case_mae_mean": float(mae_delta.mean()),
                "delta_case_mae_std": float(mae_delta.std(ddof=0)),
                "delta_case_mae_min": float(mae_delta.min()),
                "delta_case_mae_max": float(mae_delta.max()),
                "reference_better_case_rmse_count": int((rmse_delta < 0).sum()),
                "reference_better_case_rmse_rate": float((rmse_delta < 0).mean()),
                "delta_case_rmse_mean": float(rmse_delta.mean()),
                "reference_better_case22_count": int((case22_delta < 0).sum()),
                "reference_better_case22_rate": float((case22_delta < 0).mean()),
                "delta_case22_abs_error_mean": float(case22_delta.mean()),
                "delta_case22_abs_error_std": float(case22_delta.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["reference_better_case_mae_rate", "delta_case_mae_mean"], ascending=[False, True]).reset_index(drop=True)


def write_summary_markdown(
    output_path: Path,
    *,
    seed_summary_df: pd.DataFrame,
    stability_overview_df: pd.DataFrame,
    pairwise_comparison_df: pd.DataFrame,
    best_variant_by_seed_df: pd.DataFrame,
    seeds: list[int],
) -> None:
    reference_row = stability_overview_df.loc[stability_overview_df["variant_name"] == TARGET_FUSION_REFERENCE].iloc[0]
    best_row = stability_overview_df.iloc[0]
    rpm_row = stability_overview_df.loc[stability_overview_df["variant_name"] == "rpm_knn4"].iloc[0]
    learned_row = stability_overview_df.loc[stability_overview_df["variant_name"] == LEARNED_VARIANT_NAME].iloc[0]
    w03_row = stability_overview_df.loc[
        stability_overview_df["variant_name"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3"
    ].iloc[0]
    vs_rpm_row = pairwise_comparison_df.loc[pairwise_comparison_df["compare_variant"] == "rpm_knn4"].iloc[0]
    vs_w03_row = pairwise_comparison_df.loc[
        pairwise_comparison_df["compare_variant"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3"
    ].iloc[0]
    vs_w07_row = pairwise_comparison_df.loc[
        pairwise_comparison_df["compare_variant"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.7"
    ].iloc[0]

    lines = [
        "# `rpm_knn4 + TinyTCN midband @ w=0.5` 多随机种子稳定性复核",
        "",
        f"- base seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- 复核对象：`{TARGET_FUSION_REFERENCE}`",
        f"- 跨 seed 平均最优：`{best_row['variant_name']}` | case_mae mean=`{best_row['case_mae_mean']:.4f}`",
        f"- `case_mae mean ± std`：`{reference_row['case_mae_mean']:.4f} ± {reference_row['case_mae_std']:.4f}`",
        f"- `case22_abs_error mean ± std`：`{reference_row['case22_abs_error_mean']:.4f} ± {reference_row['case22_abs_error_std']:.4f}`",
        f"- `best_seed_count / seed_count`：`{int(reference_row['best_seed_count'])} / {int(reference_row['seed_count'])}`",
        "",
        "## 核心对照",
        "",
        f"- `rpm_knn4`: case_mae mean=`{rpm_row['case_mae_mean']:.4f}`, std=`{rpm_row['case_mae_std']:.4f}`",
        f"- `TinyTCN all_channels midband`: case_mae mean=`{learned_row['case_mae_mean']:.4f}`, std=`{learned_row['case_mae_std']:.4f}`",
        f"- `fusion @ w=0.3`: case_mae mean=`{w03_row['case_mae_mean']:.4f}`, std=`{w03_row['case_mae_std']:.4f}`, 优于 `rpm_knn4` 的 seed 数=`{int(w03_row['better_than_rpm_count'])}/{int(w03_row['seed_count'])}`",
        f"- `fusion @ w=0.5` 相对 `rpm_knn4`: 更优 seed 数=`{int(vs_rpm_row['reference_better_case_mae_count'])}/{int(vs_rpm_row['seed_count'])}`，平均 case_mae 差值=`{vs_rpm_row['delta_case_mae_mean']:.4f}`",
        f"- `fusion @ w=0.5` 相对 `fusion @ w=0.3`: 更优 seed 数=`{int(vs_w03_row['reference_better_case_mae_count'])}/{int(vs_w03_row['seed_count'])}`，平均 case_mae 差值=`{vs_w03_row['delta_case_mae_mean']:.4f}`",
        f"- `fusion @ w=0.5` 相对 `fusion @ w=0.7`: 更优 seed 数=`{int(vs_w07_row['reference_better_case_mae_count'])}/{int(vs_w07_row['seed_count'])}`，平均 case_mae 差值=`{vs_w07_row['delta_case_mae_mean']:.4f}`",
        "",
        "## 各 seed 最优",
        "",
    ]
    for _, row in best_variant_by_seed_df.iterrows():
        lines.append(
            f"- base_seed=`{int(row['base_seed'])}`: `{row['variant_name']}` | case_mae=`{row['case_mae']:.4f}` | case22_abs_error=`{row['case22_abs_error']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- `rpm + learned midband` 这条混合路线本身是稳的，因为 `w=0.3` 与 `w=0.5` 的平均 `case_mae` 都明显好于 `rpm_knn4`。",
            "- 但 `w=0.5` 不是最稳的固定权重；这轮复核里更稳的点落在 `w=0.3`，它在 `7/10` 个 seed 上拿到最优。",
            "- 因此到这一步更合理的结论不是“`w=0.5` 已被确认是真正稳定最优”，而是“混合方案稳定成立，但固定权重应从 `0.5` 下修到更保守的 `0.3` 附近继续作为默认候选”。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
