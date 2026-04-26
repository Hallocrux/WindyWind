from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "043_3_fft_midband_fusion_replay"
TRY042_CASE_PATH = (
    REPO_ROOT / "outputs" / "try" / "042_rpm_learned_midband_multiseed_stability_check" / "seed_case_level_predictions.csv"
)
TRY0432_CASE_PATH = (
    REPO_ROOT / "outputs" / "try" / "043_2_fft_rpm_to_wind_replay" / "fft_rpm_to_wind_case_level_predictions.csv"
)
LEARNED_VARIANT = "tinytcn_all_channels_midband_3_0_6_0hz"
REFERENCE_VARIANTS_042 = [
    "rpm_knn4",
    LEARNED_VARIANT,
    "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
    "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5",
    "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.7",
]
FFT_REFERENCE_VARIANTS = [
    "fft_fft_peak_1x_whole__to__rpm_knn4",
    "fft_hybrid_peak_1x_whole_window8_gate150__to__rpm_knn4",
    "fft_window_peak_1x_conf_8s__to__rpm_knn4",
]
FUSION_WEIGHTS = (0.3, 0.5, 0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用已有输出，回放 FFT RPM 与 learned midband 的融合表现。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "try" / TRY_NAME,
        help="输出目录，默认写入 outputs/try/043_3_fft_midband_fusion_replay/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_df = load_try042_case_level()
    seeds = sorted(seed_df["base_seed"].drop_duplicates().astype(int).tolist())
    fft_df = load_try0432_case_level(seeds)

    fusion_df = build_fusion_case_level(seed_df, fft_df)
    case_level_df = (
        pd.concat([seed_df, fft_df, fusion_df], ignore_index=True)
        .sort_values(["base_seed", "variant_name", "case_id"])
        .reset_index(drop=True)
    )
    seed_summary_df = build_seed_summary(case_level_df)
    stability_overview_df = build_stability_overview(seed_summary_df)
    candidate_variants = fusion_df["variant_name"].drop_duplicates().tolist()
    pairwise_df = build_pairwise_comparison(seed_summary_df, candidate_variants)
    best_variant_by_seed_df = build_best_variant_by_seed(seed_summary_df, candidate_variants)
    summary_md = build_summary_markdown(stability_overview_df, pairwise_df, best_variant_by_seed_df)

    case_level_df.to_csv(args.output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    seed_summary_df.to_csv(args.output_dir / "seed_summary.csv", index=False, encoding="utf-8-sig")
    stability_overview_df.to_csv(args.output_dir / "stability_overview.csv", index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(args.output_dir / "pairwise_comparison.csv", index=False, encoding="utf-8-sig")
    best_variant_by_seed_df.to_csv(args.output_dir / "best_variant_by_seed.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.md").write_text(summary_md, encoding="utf-8")


def load_try042_case_level() -> pd.DataFrame:
    df = pd.read_csv(TRY042_CASE_PATH, encoding="utf-8-sig")
    df = df.loc[df["variant_name"].isin(REFERENCE_VARIANTS_042)].copy()
    df["source_try"] = "042_rpm_learned_midband_multiseed_stability_check"
    return df.reset_index(drop=True)


def load_try0432_case_level(seeds: list[int]) -> pd.DataFrame:
    df = pd.read_csv(TRY0432_CASE_PATH, encoding="utf-8-sig")
    df = df.loc[
        (df["domain"] == "added_external")
        & (df["variant_name"].isin(FFT_REFERENCE_VARIANTS))
    ].copy()
    if df.empty:
        raise ValueError("未读取到 043_2 的 FFT added_external 回放结果。")

    expanded_blocks: list[pd.DataFrame] = []
    for base_seed in seeds:
        block = df.copy()
        block["base_seed"] = int(base_seed)
        block["seed_order"] = seeds.index(base_seed) + 1
        block["rpm"] = block["true_rpm"]
        block["model_family"] = "analytic"
        block["input_columns"] = "fft_rpm_to_wind"
        block["strain_transform"] = "none"
        block["fusion_weight_learned"] = np.nan
        block["source_try"] = "043_2_fft_rpm_to_wind_replay"
        expanded_blocks.append(
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
    return pd.concat(expanded_blocks, ignore_index=True)


def build_fusion_case_level(seed_df: pd.DataFrame, fft_df: pd.DataFrame) -> pd.DataFrame:
    learned_df = seed_df.loc[
        seed_df["variant_name"] == LEARNED_VARIANT,
        ["base_seed", "seed_order", "case_id", "file_name", "true_wind_speed", "rpm", "pred_wind_speed"],
    ].rename(columns={"pred_wind_speed": "pred_learned"})
    fft_merge_df = fft_df.loc[
        :,
        ["base_seed", "seed_order", "case_id", "file_name", "true_wind_speed", "rpm", "variant_name", "pred_wind_speed"],
    ].rename(columns={"variant_name": "fft_variant_name", "pred_wind_speed": "pred_fft"})

    merged = learned_df.merge(
        fft_merge_df,
        on=["base_seed", "seed_order", "case_id", "file_name", "true_wind_speed", "rpm"],
        how="inner",
    )
    rows: list[pd.DataFrame] = []
    for learned_weight in FUSION_WEIGHTS:
        block = merged[
            ["base_seed", "seed_order", "case_id", "file_name", "true_wind_speed", "rpm", "fft_variant_name"]
        ].copy()
        block["pred_wind_speed"] = (1.0 - learned_weight) * merged["pred_fft"] + learned_weight * merged["pred_learned"]
        block["variant_name"] = (
            "fusion_"
            + block["fft_variant_name"].str.replace("__to__", "__", regex=False)
            + f"__tinytcn_all_channels_midband__w{learned_weight:.1f}"
        )
        block["model_family"] = "fusion"
        block["input_columns"] = "fft_rpm_to_wind + all_channels_midband"
        block["strain_transform"] = "strain_bandpass_3.0_6.0Hz"
        block["fusion_weight_learned"] = learned_weight
        block["signed_error"] = block["pred_wind_speed"] - block["true_wind_speed"]
        block["abs_error"] = block["signed_error"].abs()
        block["source_try"] = "043_3_fft_midband_fusion_replay"
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def build_seed_summary(case_level_df: pd.DataFrame) -> pd.DataFrame:
    def rmse(series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        return float(np.sqrt(np.mean(np.square(values))))

    case22_df = (
        case_level_df.loc[case_level_df["case_id"] == 22, ["base_seed", "variant_name", "abs_error"]]
        .rename(columns={"abs_error": "case22_abs_error"})
    )
    summary_df = (
        case_level_df.groupby(
            ["base_seed", "seed_order", "variant_name", "model_family", "input_columns", "strain_transform", "fusion_weight_learned"],
            dropna=False,
            as_index=False,
        )
        .agg(
            case_mae=("abs_error", "mean"),
            case_rmse=("abs_error", rmse),
            mean_signed_error=("signed_error", "mean"),
            case_count=("case_id", "count"),
        )
        .merge(case22_df, on=["base_seed", "variant_name"], how="left")
        .sort_values(["variant_name", "base_seed"])
        .reset_index(drop=True)
    )
    return summary_df


def build_stability_overview(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    rpm_df = (
        seed_summary_df.loc[seed_summary_df["variant_name"] == "rpm_knn4", ["base_seed", "case_mae", "case22_abs_error"]]
        .rename(columns={"case_mae": "rpm_case_mae", "case22_abs_error": "rpm_case22_abs_error"})
    )
    merged = seed_summary_df.merge(rpm_df, on="base_seed", how="left")
    overview_df = (
        merged.groupby(
            ["variant_name", "model_family", "input_columns", "strain_transform", "fusion_weight_learned"],
            dropna=False,
            as_index=False,
        )
        .agg(
            seed_count=("base_seed", "count"),
            case_mae_mean=("case_mae", "mean"),
            case_mae_std=("case_mae", "std"),
            case_rmse_mean=("case_rmse", "mean"),
            mean_signed_error_mean=("mean_signed_error", "mean"),
            case22_abs_error_mean=("case22_abs_error", "mean"),
            case22_abs_error_std=("case22_abs_error", "std"),
            better_than_rpm_count=("case_mae", lambda s: int(np.sum(s.to_numpy(dtype=float) < merged.loc[s.index, "rpm_case_mae"].to_numpy(dtype=float)))),
            better_than_rpm_case22_count=(
                "case22_abs_error",
                lambda s: int(np.sum(s.to_numpy(dtype=float) < merged.loc[s.index, "rpm_case22_abs_error"].to_numpy(dtype=float))),
            ),
        )
        .sort_values(["case_mae_mean", "case22_abs_error_mean", "variant_name"])
        .reset_index(drop=True)
    )
    overview_df["better_than_rpm_rate"] = overview_df["better_than_rpm_count"] / overview_df["seed_count"]
    overview_df["better_than_rpm_case22_rate"] = overview_df["better_than_rpm_case22_count"] / overview_df["seed_count"]
    return overview_df


def build_pairwise_comparison(seed_summary_df: pd.DataFrame, candidate_variants: list[str]) -> pd.DataFrame:
    reference_variants = [
        "rpm_knn4",
        LEARNED_VARIANT,
        "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
        "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5",
        "fft_fft_peak_1x_whole__to__rpm_knn4",
        "fft_hybrid_peak_1x_whole_window8_gate150__to__rpm_knn4",
        "fft_window_peak_1x_conf_8s__to__rpm_knn4",
    ]
    rows: list[dict[str, object]] = []
    for candidate_variant in candidate_variants:
        candidate_df = seed_summary_df.loc[
            seed_summary_df["variant_name"] == candidate_variant,
            ["base_seed", "case_mae", "case22_abs_error"],
        ].rename(columns={"case_mae": "candidate_case_mae", "case22_abs_error": "candidate_case22_abs_error"})
        for reference_variant in reference_variants:
            reference_df = seed_summary_df.loc[
                seed_summary_df["variant_name"] == reference_variant,
                ["base_seed", "case_mae", "case22_abs_error"],
            ].rename(columns={"case_mae": "reference_case_mae", "case22_abs_error": "reference_case22_abs_error"})
            merged = candidate_df.merge(reference_df, on="base_seed", how="inner")
            if merged.empty:
                continue
            rows.append(
                {
                    "candidate_variant": candidate_variant,
                    "compare_variant": reference_variant,
                    "seed_count": int(len(merged)),
                    "candidate_better_case_mae_count": int(
                        np.sum(merged["candidate_case_mae"].to_numpy(dtype=float) < merged["reference_case_mae"].to_numpy(dtype=float))
                    ),
                    "candidate_better_case22_count": int(
                        np.sum(
                            merged["candidate_case22_abs_error"].to_numpy(dtype=float)
                            < merged["reference_case22_abs_error"].to_numpy(dtype=float)
                        )
                    ),
                    "delta_case_mae_mean": float(
                        np.mean(merged["candidate_case_mae"].to_numpy(dtype=float) - merged["reference_case_mae"].to_numpy(dtype=float))
                    ),
                    "delta_case22_abs_error_mean": float(
                        np.mean(
                            merged["candidate_case22_abs_error"].to_numpy(dtype=float)
                            - merged["reference_case22_abs_error"].to_numpy(dtype=float)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["candidate_variant", "delta_case_mae_mean", "compare_variant"]).reset_index(drop=True)


def build_best_variant_by_seed(seed_summary_df: pd.DataFrame, candidate_variants: list[str]) -> pd.DataFrame:
    return (
        seed_summary_df.loc[seed_summary_df["variant_name"].isin(candidate_variants)]
        .sort_values(["base_seed", "case_mae", "case22_abs_error", "variant_name"])
        .groupby("base_seed", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def build_summary_markdown(
    stability_overview_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    best_variant_by_seed_df: pd.DataFrame,
) -> str:
    candidate_df = stability_overview_df.loc[stability_overview_df["variant_name"].str.startswith("fusion_fft_")].copy()
    best_row = candidate_df.iloc[0]
    true_upper_row = stability_overview_df.loc[
        stability_overview_df["variant_name"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3"
    ].iloc[0]
    fft_base_row = stability_overview_df.loc[
        stability_overview_df["variant_name"] == "fft_fft_peak_1x_whole__to__rpm_knn4"
    ].iloc[0]
    best_vs_true = pairwise_df.loc[
        (pairwise_df["candidate_variant"] == best_row["variant_name"])
        & (pairwise_df["compare_variant"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3")
    ].iloc[0]
    best_vs_fft = pairwise_df.loc[
        (pairwise_df["candidate_variant"] == best_row["variant_name"])
        & (pairwise_df["compare_variant"] == "fft_fft_peak_1x_whole__to__rpm_knn4")
    ].iloc[0]

    lines = [
        "# FFT RPM 与 learned midband 融合回放",
        "",
        "## 当前结果",
        "",
        f"- 最优 deployable 融合候选：`{best_row['variant_name']}`",
        f"- `case_mae mean = {best_row['case_mae_mean']:.4f}`",
        f"- `case_mae std = {best_row['case_mae_std']:.4f}`",
        f"- `case22_abs_error mean = {best_row['case22_abs_error_mean']:.4f}`",
        "",
        "## 关键对照",
        "",
        f"- `true_rpm` 上界参考 `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3`: case_mae mean=`{true_upper_row['case_mae_mean']:.4f}`",
        f"- FFT 单独可部署基线 `fft_fft_peak_1x_whole__to__rpm_knn4`: case_mae mean=`{fft_base_row['case_mae_mean']:.4f}`",
        f"- 最优 deployable 融合相对 `true_rpm` 上界：delta_case_mae_mean=`{best_vs_true['delta_case_mae_mean']:+.4f}`，更优 seed 数=`{int(best_vs_true['candidate_better_case_mae_count'])}/{int(best_vs_true['seed_count'])}`",
        f"- 最优 deployable 融合相对 FFT 单独基线：delta_case_mae_mean=`{best_vs_fft['delta_case_mae_mean']:+.4f}`，更优 seed 数=`{int(best_vs_fft['candidate_better_case_mae_count'])}/{int(best_vs_fft['seed_count'])}`",
        "",
        "## 当前判断",
        "",
        "- FFT 支线已经可以替代 `true_rpm` 支线，进入 added 方向的可部署融合链。",
        "- 这个“替代”更准确地说是“接近上界的 deployable 替代”，不是与 `true_rpm` 完全等价。",
        "- 当前更稳的默认 deployable 组合应优先参考：`fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.3`。",
        "",
        "## 每个 seed 的最优候选",
        "",
    ]
    for _, row in best_variant_by_seed_df.iterrows():
        lines.append(
            f"- seed `{int(row['base_seed'])}`: `{row['variant_name']}` | case_mae=`{row['case_mae']:.4f}` | case22_abs_error=`{row['case22_abs_error']:.4f}`"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
