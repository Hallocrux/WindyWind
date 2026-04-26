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


TRY_NAME = "044_final_loco_fft_midband_fusion_check"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
FFT_CASE_PATH = REPO_ROOT / "outputs" / "try" / "043_2_fft_rpm_to_wind_replay" / "fft_rpm_to_wind_case_level_predictions.csv"
DEFAULT_BASE_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
LEARNED_VARIANT_NAME = "tinytcn_all_channels_midband_3_0_6_0hz_final_loco"
TRUE_RPM_VARIANT_NAME = "true_rpm_knn4_final_loco"
FFT_REFERENCE_VARIANTS = [
    "fft_window_peak_1x_conf_8s__to__rpm_knn4",
    "fft_hybrid_peak_1x_whole_window8_gate150__to__rpm_knn4",
    "fft_fft_peak_1x_whole__to__rpm_knn4",
]
FUSION_WEIGHTS = (0.3, 0.5, 0.7)
TINYTCN_SEED_OFFSET = 53 * 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补齐 final LOCO 下的 FFT + learned midband 融合验证。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_BASE_SEEDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_run_dir = args.output_dir / "seed_runs"
    seed_run_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)

    train_config = try041.TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    final_records = [record for record in try041.scan_dataset_records() if record.is_labeled]
    common_signal_columns = try041.get_common_signal_columns(final_records)
    strain_columns = [column for column in common_signal_columns if "应变" in column]
    acc_columns = [column for column in common_signal_columns if "Acc" in column]
    selected_columns = [*strain_columns, *acc_columns]

    base_frames = {
        record.case_id: try041.load_clean_signal_frame(record, common_signal_columns)
        for record in final_records
    }
    midband_frames = try041.build_midband_frames(base_frames, final_records, strain_columns)
    fft_case_df = load_fft_final_loco_case_df(args.seeds)

    seed_case_rows: list[pd.DataFrame] = []
    seed_summary_rows: list[pd.DataFrame] = []
    for seed_order, base_seed in enumerate(args.seeds, start=1):
        tinytcn_seed = base_seed + TINYTCN_SEED_OFFSET
        seed_case_path = seed_run_dir / f"seed_{base_seed}_case_level_predictions.csv"
        seed_summary_path = seed_run_dir / f"seed_{base_seed}_summary.csv"
        if seed_case_path.exists() and seed_summary_path.exists():
            print(f"[reuse] seed={base_seed}")
            seed_case_df = pd.read_csv(seed_case_path, encoding="utf-8-sig")
            seed_summary_df = pd.read_csv(seed_summary_path, encoding="utf-8-sig")
        else:
            print(f"[run] seed={base_seed}")
            seed_case_df = run_single_seed(
                base_seed=base_seed,
                seed_order=seed_order,
                tinytcn_seed=tinytcn_seed,
                final_records=final_records,
                midband_frames=midband_frames,
                selected_columns=selected_columns,
                train_config=train_config,
                fft_case_df=fft_case_df.loc[fft_case_df["base_seed"] == base_seed].copy(),
            )
            seed_summary_df = build_seed_summary(seed_case_df)
            seed_summary_df.insert(0, "seed_order", seed_order)
            seed_summary_df.insert(1, "base_seed", base_seed)
            seed_summary_df.insert(2, "tinytcn_seed", tinytcn_seed)
            seed_case_df.to_csv(seed_case_path, index=False, encoding="utf-8-sig")
            seed_summary_df.to_csv(seed_summary_path, index=False, encoding="utf-8-sig")
            print(f"[done] seed={base_seed}")
        seed_case_rows.append(seed_case_df)
        seed_summary_rows.append(seed_summary_df)

    case_level_df = pd.concat(seed_case_rows, ignore_index=True)
    seed_summary_df = pd.concat(seed_summary_rows, ignore_index=True)
    seed_summary_df = add_seed_ranks(seed_summary_df)
    best_variant_by_seed_df = build_best_variant_by_seed(seed_summary_df)
    stability_overview_df = build_stability_overview(seed_summary_df)
    pairwise_comparison_df = build_pairwise_comparison(seed_summary_df)
    summary_md = build_summary_markdown(stability_overview_df, pairwise_comparison_df, best_variant_by_seed_df)

    case_level_df.to_csv(args.output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    seed_summary_df.to_csv(args.output_dir / "seed_summary.csv", index=False, encoding="utf-8-sig")
    stability_overview_df.to_csv(args.output_dir / "stability_overview.csv", index=False, encoding="utf-8-sig")
    pairwise_comparison_df.to_csv(args.output_dir / "pairwise_comparison.csv", index=False, encoding="utf-8-sig")
    best_variant_by_seed_df.to_csv(args.output_dir / "best_variant_by_seed.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.md").write_text(summary_md, encoding="utf-8")


def load_fft_final_loco_case_df(seeds: list[int]) -> pd.DataFrame:
    df = pd.read_csv(FFT_CASE_PATH, encoding="utf-8-sig")
    df = df.loc[
        (df["domain"] == "final_loco")
        & (df["variant_name"].isin(FFT_REFERENCE_VARIANTS))
    ].copy()
    blocks: list[pd.DataFrame] = []
    for seed_order, base_seed in enumerate(seeds, start=1):
        block = df.copy()
        block["base_seed"] = base_seed
        block["seed_order"] = seed_order
        block["rpm"] = block["true_rpm"]
        block["model_family"] = "analytic"
        block["input_columns"] = "fft_rpm_to_wind"
        block["strain_transform"] = "none"
        block["fusion_weight_learned"] = np.nan
        block["source_try"] = "043_2_fft_rpm_to_wind_replay"
        blocks.append(
            block[
                [
                    "base_seed",
                    "seed_order",
                    "case_id",
                    "file_name",
                    "true_wind_speed",
                    "rpm",
                    "pred_wind_speed",
                    "variant_name",
                    "model_family",
                    "input_columns",
                    "strain_transform",
                    "fusion_weight_learned",
                    "signed_error",
                    "abs_error",
                    "source_try",
                ]
            ]
        )
    return pd.concat(blocks, ignore_index=True)


def run_single_seed(
    *,
    base_seed: int,
    seed_order: int,
    tinytcn_seed: int,
    final_records: list[try041.DatasetRecord],
    midband_frames: dict[int, pd.DataFrame],
    selected_columns: list[str],
    train_config: try041.TrainConfig,
    fft_case_df: pd.DataFrame,
) -> pd.DataFrame:
    learned_df = build_final_loco_learned_predictions(
        final_records=final_records,
        midband_frames=midband_frames,
        selected_columns=selected_columns,
        train_config=train_config,
        seed=tinytcn_seed,
    )
    true_rpm_df = build_true_rpm_final_loco_predictions(final_records)

    base_case_df = pd.concat([learned_df, true_rpm_df, fft_case_df], ignore_index=True)
    base_case_df["seed_order"] = seed_order
    base_case_df["base_seed"] = base_seed
    base_case_df["tinytcn_seed"] = tinytcn_seed
    fusion_df = build_fusion_predictions(base_case_df)
    case_level_df = pd.concat([base_case_df, fusion_df], ignore_index=True)
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()
    return case_level_df.sort_values(["variant_name", "case_id"]).reset_index(drop=True)


def build_final_loco_learned_predictions(
    *,
    final_records: list[try041.DatasetRecord],
    midband_frames: dict[int, pd.DataFrame],
    selected_columns: list[str],
    train_config: try041.TrainConfig,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for holdout in final_records:
        train_records = [record for record in final_records if record.case_id != holdout.case_id]
        eval_records = [holdout]
        pred_df = try041.train_and_predict_tinytcn(
            train_records=train_records,
            eval_records=eval_records,
            cleaned_signal_frames=midband_frames,
            selected_columns=selected_columns,
            train_config=train_config,
            seed=seed,
        )
        pred_df["variant_name"] = LEARNED_VARIANT_NAME
        pred_df["model_family"] = "tinytcn"
        pred_df["input_columns"] = "all_channels_midband"
        pred_df["strain_transform"] = "strain_bandpass_3.0_6.0Hz"
        pred_df["fusion_weight_learned"] = np.nan
        pred_df["source_try"] = TRY_NAME
        rows.append(pred_df)
    return pd.concat(rows, ignore_index=True)


def build_true_rpm_final_loco_predictions(final_records: list[try041.DatasetRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    final_manifest_df = pd.read_csv(REPO_ROOT / "data" / "final" / "dataset_manifest.csv")
    final_manifest_df["wind_speed"] = pd.to_numeric(final_manifest_df["wind_speed"], errors="coerce")
    final_manifest_df["rpm"] = pd.to_numeric(final_manifest_df["rpm"], errors="coerce")
    final_manifest_df = final_manifest_df.dropna(subset=["wind_speed", "rpm"]).copy()
    for holdout in final_records:
        train_df = final_manifest_df.loc[final_manifest_df["case_id"] != holdout.case_id].copy()
        nearest_df = train_df.assign(rpm_distance=(train_df["rpm"] - float(holdout.rpm)).abs()).nsmallest(4, "rpm_distance")
        pred = try041.weighted_rpm_neighbor_prediction(nearest_df)
        rows.append(
            {
                "case_id": holdout.case_id,
                "file_name": holdout.file_name,
                "true_wind_speed": float(holdout.wind_speed),
                "rpm": float(holdout.rpm),
                "pred_wind_speed": pred,
                "variant_name": TRUE_RPM_VARIANT_NAME,
                "model_family": "analytic",
                "input_columns": "true_rpm",
                "strain_transform": "none",
                "fusion_weight_learned": np.nan,
                "source_try": TRY_NAME,
            }
        )
    return pd.DataFrame(rows)


def build_fusion_predictions(case_level_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["case_id", "file_name", "true_wind_speed", "rpm", "base_seed", "seed_order", "tinytcn_seed"]
    learned_df = case_level_df.loc[
        case_level_df["variant_name"] == LEARNED_VARIANT_NAME,
        base_columns + ["pred_wind_speed"],
    ].rename(columns={"pred_wind_speed": "pred_learned"})

    rows: list[pd.DataFrame] = []

    true_rpm_df = case_level_df.loc[
        case_level_df["variant_name"] == TRUE_RPM_VARIANT_NAME,
        base_columns + ["pred_wind_speed"],
    ].rename(columns={"pred_wind_speed": "pred_rpm_like"})
    merged_true = true_rpm_df.merge(learned_df, on=base_columns, how="inner")
    rows.extend(build_weighted_blocks(merged_true, "fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco"))

    fft_df = case_level_df.loc[
        case_level_df["variant_name"].isin(FFT_REFERENCE_VARIANTS),
        base_columns + ["variant_name", "pred_wind_speed"],
    ].rename(columns={"variant_name": "fft_variant_name", "pred_wind_speed": "pred_rpm_like"})
    for fft_variant_name, block in fft_df.groupby("fft_variant_name", sort=True):
        merged_fft = block.merge(learned_df, on=base_columns, how="inner")
        prefix = "fusion_" + fft_variant_name.replace("__to__", "__")
        rows.extend(build_weighted_blocks(merged_fft, prefix))

    return pd.concat(rows, ignore_index=True)


def build_weighted_blocks(merged_df: pd.DataFrame, variant_prefix: str) -> list[pd.DataFrame]:
    base_columns = ["case_id", "file_name", "true_wind_speed", "rpm", "base_seed", "seed_order", "tinytcn_seed"]
    rows: list[pd.DataFrame] = []
    for learned_weight in FUSION_WEIGHTS:
        block = merged_df[base_columns].copy()
        block["pred_wind_speed"] = (1.0 - learned_weight) * merged_df["pred_rpm_like"] + learned_weight * merged_df["pred_learned"]
        block["variant_name"] = f"{variant_prefix}__w{learned_weight:.1f}"
        block["model_family"] = "fusion"
        block["input_columns"] = "rpm_like + all_channels_midband"
        block["strain_transform"] = "strain_bandpass_3.0_6.0Hz"
        block["fusion_weight_learned"] = learned_weight
        block["source_try"] = TRY_NAME
        rows.append(block)
    return rows


def build_seed_summary(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_level_df.groupby("variant_name", sort=False):
        rows.append(
            {
                "variant_name": variant_name,
                "model_family": block["model_family"].iloc[0],
                "input_columns": block["input_columns"].iloc[0],
                "strain_transform": block["strain_transform"].iloc[0],
                "fusion_weight_learned": block["fusion_weight_learned"].iloc[0],
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case1_abs_error": float(block.loc[block["case_id"] == 1, "abs_error"].iloc[0]),
                "case18_abs_error": float(block.loc[block["case_id"] == 18, "abs_error"].iloc[0]),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_mae", "case1_abs_error", "variant_name"]).reset_index(drop=True)


def add_seed_ranks(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked_blocks: list[pd.DataFrame] = []
    for base_seed, block in seed_summary_df.groupby("base_seed", sort=True):
        ordered = block.sort_values(["case_mae", "case1_abs_error", "variant_name"]).reset_index(drop=True).copy()
        ordered["case_mae_rank"] = np.arange(1, len(ordered) + 1)
        ranked_blocks.append(ordered)
    return pd.concat(ranked_blocks, ignore_index=True)


def build_best_variant_by_seed(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_summary_df.loc[seed_summary_df["case_mae_rank"] == 1]
        .sort_values(["base_seed", "case_mae", "variant_name"])
        .reset_index(drop=True)
    )


def build_stability_overview(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    fft_best_df = seed_summary_df.loc[
        seed_summary_df["variant_name"] == "fft_window_peak_1x_conf_8s__to__rpm_knn4",
        ["base_seed", "case_mae", "case1_abs_error"],
    ].rename(columns={"case_mae": "fft_best_case_mae", "case1_abs_error": "fft_best_case1_abs_error"})
    rows: list[dict[str, object]] = []
    for variant_name, block in seed_summary_df.groupby("variant_name", sort=False):
        merged_vs_fft = block[["base_seed", "case_mae", "case1_abs_error"]].merge(fft_best_df, on="base_seed", how="inner")
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
                "case_rmse_mean": float(block["case_rmse"].mean()),
                "mean_signed_error_mean": float(block["mean_signed_error"].mean()),
                "case1_abs_error_mean": float(block["case1_abs_error"].mean()),
                "case18_abs_error_mean": float(block["case18_abs_error"].mean()),
                "better_than_fft_best_count": int(np.sum(merged_vs_fft["case_mae"] < merged_vs_fft["fft_best_case_mae"])),
                "delta_vs_fft_best_case_mae_mean": float(np.mean(merged_vs_fft["case_mae"] - merged_vs_fft["fft_best_case_mae"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_mae_mean", "variant_name"]).reset_index(drop=True)


def build_pairwise_comparison(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    candidate_variants = [
        variant_name
        for variant_name in seed_summary_df["variant_name"].unique().tolist()
        if variant_name.startswith("fusion_fft_")
    ]
    reference_variants = [
        LEARNED_VARIANT_NAME,
        TRUE_RPM_VARIANT_NAME,
        "fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco__w0.3",
        "fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco__w0.5",
        "fft_window_peak_1x_conf_8s__to__rpm_knn4",
        "fft_hybrid_peak_1x_whole_window8_gate150__to__rpm_knn4",
        "fft_fft_peak_1x_whole__to__rpm_knn4",
    ]
    rows: list[dict[str, object]] = []
    for candidate_variant in candidate_variants:
        candidate_df = seed_summary_df.loc[
            seed_summary_df["variant_name"] == candidate_variant,
            ["base_seed", "case_mae", "case1_abs_error", "case18_abs_error"],
        ].rename(
            columns={
                "case_mae": "candidate_case_mae",
                "case1_abs_error": "candidate_case1_abs_error",
                "case18_abs_error": "candidate_case18_abs_error",
            }
        )
        for reference_variant in reference_variants:
            reference_df = seed_summary_df.loc[
                seed_summary_df["variant_name"] == reference_variant,
                ["base_seed", "case_mae", "case1_abs_error", "case18_abs_error"],
            ].rename(
                columns={
                    "case_mae": "reference_case_mae",
                    "case1_abs_error": "reference_case1_abs_error",
                    "case18_abs_error": "reference_case18_abs_error",
                }
            )
            merged = candidate_df.merge(reference_df, on="base_seed", how="inner")
            if merged.empty:
                continue
            rows.append(
                {
                    "candidate_variant": candidate_variant,
                    "compare_variant": reference_variant,
                    "seed_count": int(len(merged)),
                    "candidate_better_case_mae_count": int(np.sum(merged["candidate_case_mae"] < merged["reference_case_mae"])),
                    "candidate_better_case1_count": int(
                        np.sum(merged["candidate_case1_abs_error"] < merged["reference_case1_abs_error"])
                    ),
                    "candidate_better_case18_count": int(
                        np.sum(merged["candidate_case18_abs_error"] < merged["reference_case18_abs_error"])
                    ),
                    "delta_case_mae_mean": float(np.mean(merged["candidate_case_mae"] - merged["reference_case_mae"])),
                    "delta_case1_abs_error_mean": float(
                        np.mean(merged["candidate_case1_abs_error"] - merged["reference_case1_abs_error"])
                    ),
                    "delta_case18_abs_error_mean": float(
                        np.mean(merged["candidate_case18_abs_error"] - merged["reference_case18_abs_error"])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["candidate_variant", "delta_case_mae_mean", "compare_variant"]).reset_index(drop=True)


def build_summary_markdown(
    stability_overview_df: pd.DataFrame,
    pairwise_comparison_df: pd.DataFrame,
    best_variant_by_seed_df: pd.DataFrame,
) -> str:
    best_fft_fusion_row = stability_overview_df.loc[
        stability_overview_df["variant_name"].str.startswith("fusion_fft_")
    ].iloc[0]
    true_upper_row = stability_overview_df.loc[
        stability_overview_df["variant_name"] == "fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco__w0.3"
    ].iloc[0]
    fft_best_row = stability_overview_df.loc[
        stability_overview_df["variant_name"] == "fft_window_peak_1x_conf_8s__to__rpm_knn4"
    ].iloc[0]
    best_vs_true = pairwise_comparison_df.loc[
        (pairwise_comparison_df["candidate_variant"] == best_fft_fusion_row["variant_name"])
        & (pairwise_comparison_df["compare_variant"] == "fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco__w0.3")
    ].iloc[0]
    best_vs_fft = pairwise_comparison_df.loc[
        (pairwise_comparison_df["candidate_variant"] == best_fft_fusion_row["variant_name"])
        & (pairwise_comparison_df["compare_variant"] == "fft_window_peak_1x_conf_8s__to__rpm_knn4")
    ].iloc[0]

    lines = [
        "# final LOCO FFT + learned midband 融合补齐",
        "",
        "## 当前结果",
        "",
        f"- 最优 final LOCO FFT 融合候选：`{best_fft_fusion_row['variant_name']}`",
        f"- `case_mae mean = {best_fft_fusion_row['case_mae_mean']:.4f}`",
        f"- `case_mae std = {best_fft_fusion_row['case_mae_std']:.4f}`",
        f"- `case1_abs_error mean = {best_fft_fusion_row['case1_abs_error_mean']:.4f}`",
        f"- `case18_abs_error mean = {best_fft_fusion_row['case18_abs_error_mean']:.4f}`",
        "",
        "## 关键对照",
        "",
        f"- `true_rpm + learned midband @ w=0.3`: case_mae mean=`{true_upper_row['case_mae_mean']:.4f}`",
        f"- FFT 单独 final LOCO 基线 `fft_window_peak_1x_conf_8s__to__rpm_knn4`: case_mae mean=`{fft_best_row['case_mae_mean']:.4f}`",
        f"- 最优 FFT 融合相对 `true_rpm` 上界：delta_case_mae_mean=`{best_vs_true['delta_case_mae_mean']:+.4f}`，更优 seed 数=`{int(best_vs_true['candidate_better_case_mae_count'])}/{int(best_vs_true['seed_count'])}`",
        f"- 最优 FFT 融合相对 FFT 单独基线：delta_case_mae_mean=`{best_vs_fft['delta_case_mae_mean']:+.4f}`，更优 seed 数=`{int(best_vs_fft['candidate_better_case_mae_count'])}/{int(best_vs_fft['seed_count'])}`",
        "",
        "## 每个 seed 的最优 FFT 融合候选",
        "",
    ]
    for _, row in best_variant_by_seed_df.loc[
        best_variant_by_seed_df["variant_name"].str.startswith("fusion_fft_")
    ].iterrows():
        lines.append(
            f"- seed `{int(row['base_seed'])}`: `{row['variant_name']}` | case_mae=`{row['case_mae']:.4f}` | case1_abs_error=`{row['case1_abs_error']:.4f}`"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
