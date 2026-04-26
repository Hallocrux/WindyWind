from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "068_domain_lodo_moe_diagnosis"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_REMOTE_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "067_reuse_embedding_domain_lodo_moe"
    / "latest_kernel_output_v3"
    / "windywind"
    / "outputs"
    / "try"
    / "067_reuse_embedding_domain_lodo_moe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 067 Kaggle full run 的域行为。")
    parser.add_argument("--remote-output-dir", type=Path, default=DEFAULT_REMOTE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    case_level_df = pd.read_csv(args.remote_output_dir / "case_level_predictions.csv", encoding="utf-8-sig")
    router_df = pd.read_csv(args.remote_output_dir / "router_activation_table.csv", encoding="utf-8-sig")
    prototype_df = pd.read_csv(args.remote_output_dir / "prototype_retrieval_stats.csv", encoding="utf-8-sig")
    fold_df = pd.read_csv(args.remote_output_dir / "fold_metadata.csv", encoding="utf-8-sig")

    case_delta_df = build_case_delta_table(case_level_df)
    domain_behavior_df = build_domain_behavior_summary(case_delta_df, router_df)
    router_summary_df = build_router_summary_by_domain(router_df)
    prototype_mix_df = build_prototype_neighbor_mix_by_domain(prototype_df)
    key_failure_df = build_key_failure_cases(case_delta_df)

    case_delta_df.to_csv(output_dir / "case_delta_table.csv", index=False, encoding="utf-8-sig")
    domain_behavior_df.to_csv(output_dir / "domain_behavior_summary.csv", index=False, encoding="utf-8-sig")
    router_summary_df.to_csv(output_dir / "router_summary_by_domain.csv", index=False, encoding="utf-8-sig")
    prototype_mix_df.to_csv(output_dir / "prototype_neighbor_mix_by_domain.csv", index=False, encoding="utf-8-sig")
    key_failure_df.to_csv(output_dir / "key_failure_cases.csv", index=False, encoding="utf-8-sig")

    create_domain_improvement_plot(case_delta_df, plot_dir / "domain_case_improvement.png")
    create_router_weight_plot(router_summary_df, plot_dir / "router_weight_by_domain.png")

    write_summary_markdown(
        output_dir / "summary.md",
        domain_behavior_df=domain_behavior_df,
        router_summary_df=router_summary_df,
        prototype_mix_df=prototype_mix_df,
        key_failure_df=key_failure_df,
        fold_df=fold_df,
    )

    print("068 domain LODO MoE diagnosis 已完成。")
    print(f"输出目录: {output_dir}")


def build_case_delta_table(case_level_df: pd.DataFrame) -> pd.DataFrame:
    base_df = case_level_df.loc[case_level_df["variant_name"] == "A0_rpm_knn4"].copy()
    moe_df = case_level_df.loc[case_level_df["variant_name"] == "A3_sparse_router_moe"].copy()
    merged = (
        base_df.merge(
            moe_df,
            on=["fold_tag", "case_id", "raw_source_domain", "learned_domain_id", "learned_domain_name"],
            suffixes=("_base", "_moe"),
        )
        .sort_values(["learned_domain_id", "case_id"])
        .reset_index(drop=True)
    )
    merged["signed_error_shift"] = merged["signed_error_moe"] - merged["signed_error_base"]
    merged["abs_error_gain"] = merged["abs_error_base"] - merged["abs_error_moe"]
    merged["correction_delta"] = merged["pred_wind_speed_moe"] - merged["pred_wind_speed_base"]
    merged["baseline_underpredict"] = (merged["signed_error_base"] < 0).astype(int)
    merged["baseline_overpredict"] = (merged["signed_error_base"] > 0).astype(int)
    merged["moe_better"] = (merged["abs_error_gain"] > 0).astype(int)
    return merged[[
        "fold_tag",
        "case_id",
        "raw_source_domain",
        "learned_domain_id",
        "learned_domain_name",
        "true_wind_speed_base",
        "rpm_base",
        "pred_wind_speed_base",
        "pred_wind_speed_moe",
        "signed_error_base",
        "signed_error_moe",
        "abs_error_base",
        "abs_error_moe",
        "correction_delta",
        "signed_error_shift",
        "abs_error_gain",
        "baseline_underpredict",
        "baseline_overpredict",
        "moe_better",
    ]]


def build_domain_behavior_summary(case_delta_df: pd.DataFrame, router_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for learned_domain_id, block in case_delta_df.groupby("learned_domain_id", sort=True):
        router_block = router_df.loc[router_df["learned_domain_id"] == learned_domain_id].copy()
        rows.append(
            {
                "learned_domain_id": int(learned_domain_id),
                "learned_domain_name": str(block["learned_domain_name"].iloc[0]),
                "case_count": int(len(block)),
                "raw_source_mix": ",".join(
                    f"{domain}:{int(count)}" for domain, count in block["raw_source_domain"].value_counts().sort_index().items()
                ),
                "base_case_mae": float(block["abs_error_base"].mean()),
                "moe_case_mae": float(block["abs_error_moe"].mean()),
                "mean_abs_error_gain": float(block["abs_error_gain"].mean()),
                "better_case_rate": float(block["moe_better"].mean()),
                "base_mean_signed_error": float(block["signed_error_base"].mean()),
                "moe_mean_signed_error": float(block["signed_error_moe"].mean()),
                "mean_correction_delta": float(block["correction_delta"].mean()),
                "underpredict_rate_base": float(block["baseline_underpredict"].mean()),
                "overpredict_rate_base": float(block["baseline_overpredict"].mean()),
                "dominant_expert_mode": dominant_mode(router_block["dominant_expert"].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("learned_domain_id").reset_index(drop=True)


def build_router_summary_by_domain(router_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for learned_domain_id, block in router_df.groupby("learned_domain_id", sort=True):
        rows.append(
            {
                "learned_domain_id": int(learned_domain_id),
                "learned_domain_name": str(block["learned_domain_name"].iloc[0]),
                "case_count": int(len(block)),
                "mean_weight_noop": float(block["weight_expert0_noop"].mean()),
                "mean_weight_global": float(block["weight_expert1_global"].mean()),
                "mean_weight_prototype": float(block["weight_expert2_prototype"].mean()),
                "dominant_noop_rate": float((block["dominant_expert"] == 0).mean()),
                "dominant_global_rate": float((block["dominant_expert"] == 1).mean()),
                "dominant_prototype_rate": float((block["dominant_expert"] == 2).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("learned_domain_id").reset_index(drop=True)


def build_prototype_neighbor_mix_by_domain(prototype_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for learned_domain_id, block in prototype_df.groupby("target_learned_domain_id", sort=True):
        domain_counts = block["neighbor_learned_domain_name"].value_counts(normalize=True).sort_index()
        row = {
            "learned_domain_id": int(learned_domain_id),
            "learned_domain_name": str(block["target_learned_domain_name"].iloc[0]),
            "mean_same_domain_rate_reported": float(block["same_domain_rate"].mean()),
            "mean_top1_distance": float(block["top1_distance"].mean()),
            "mean_topk_distance": float(block["topk_mean_distance"].mean()),
        }
        for domain_name, ratio in domain_counts.items():
            row[f"neighbor_ratio__{domain_name}"] = float(ratio)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("learned_domain_id").reset_index(drop=True)


def build_key_failure_cases(case_delta_df: pd.DataFrame) -> pd.DataFrame:
    failure_df = case_delta_df.loc[case_delta_df["abs_error_gain"] < 0].copy()
    return failure_df.sort_values("abs_error_gain").reset_index(drop=True)


def dominant_mode(values: list[int]) -> str:
    if not values:
        return "unknown"
    mapping = {0: "noop", 1: "global", 2: "prototype"}
    series = pd.Series(values).value_counts()
    return mapping.get(int(series.index[0]), f"expert_{int(series.index[0])}")


def create_domain_improvement_plot(case_delta_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = case_delta_df.copy()
    colors = ["#2a9d8f" if value > 0 else "#d94841" for value in plot_df["abs_error_gain"]]
    labels = [f"d{int(row.learned_domain_id)}-c{int(row.case_id)}" for row in plot_df.itertuples()]
    ax.bar(np.arange(len(plot_df)), plot_df["abs_error_gain"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("abs_error_gain (A0 - A3)")
    ax.set_title("Case-level improvement by learned domain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_router_weight_plot(router_summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(router_summary_df))
    width = 0.25
    ax.bar(x - width, router_summary_df["mean_weight_noop"], width=width, label="noop", color="#7f8c8d")
    ax.bar(x, router_summary_df["mean_weight_global"], width=width, label="global", color="#386cb0")
    ax.bar(x + width, router_summary_df["mean_weight_prototype"], width=width, label="prototype", color="#fdb462")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{int(v)}" for v in router_summary_df["learned_domain_id"]], fontsize=10)
    ax.set_ylabel("mean router weight")
    ax.set_title("Router behavior by learned domain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    *,
    domain_behavior_df: pd.DataFrame,
    router_summary_df: pd.DataFrame,
    prototype_mix_df: pd.DataFrame,
    key_failure_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    domain0 = domain_behavior_df.loc[domain_behavior_df["learned_domain_id"] == 0].iloc[0]
    domain2 = domain_behavior_df.loc[domain_behavior_df["learned_domain_id"] == 2].iloc[0]
    domain3 = domain_behavior_df.loc[domain_behavior_df["learned_domain_id"] == 3].iloc[0]
    router0 = router_summary_df.loc[router_summary_df["learned_domain_id"] == 0].iloc[0]
    router2 = router_summary_df.loc[router_summary_df["learned_domain_id"] == 2].iloc[0]
    router3 = router_summary_df.loc[router_summary_df["learned_domain_id"] == 3].iloc[0]
    proto0 = prototype_mix_df.loc[prototype_mix_df["learned_domain_id"] == 0].iloc[0]

    lines = [
        "# domain LODO MoE diagnosis",
        "",
        "## 总览",
        "",
        f"- `domain_0`：`A3` 相比 `A0` 的 mean_abs_error_gain = `{float(domain0['mean_abs_error_gain']):.4f}`，better_case_rate = `{float(domain0['better_case_rate']):.4f}`",
        f"- `domain_2`：`A3` 相比 `A0` 的 mean_abs_error_gain = `{float(domain2['mean_abs_error_gain']):.4f}`，better_case_rate = `{float(domain2['better_case_rate']):.4f}`",
        f"- `domain_3`：`A3` 相比 `A0` 的 mean_abs_error_gain = `{float(domain3['mean_abs_error_gain']):.4f}`，better_case_rate = `{float(domain3['better_case_rate']):.4f}`",
        "",
        "## 诊断结论",
        "",
        (
            f"- `domain_0` 的正信号主要来自“系统性低估 + correction 方向一致”："
            f"`A0` 的 mean_signed_error = `{float(domain0['base_mean_signed_error']):.4f}`，"
            f"`A3` 的 mean_correction_delta = `{float(domain0['mean_correction_delta']):.4f}`，"
            f"且 better_case_rate = `{float(domain0['better_case_rate']):.4f}`。"
        ),
        (
            f"- `domain_0` 上 router 完全偏向 `prototype`："
            f"dominant_prototype_rate = `{float(router0['dominant_prototype_rate']):.4f}`；"
            f"reported same_domain_rate = `{float(proto0['mean_same_domain_rate_reported']):.4f}`。"
        ),
        (
            f"- `domain_2` 的退化主要来自 `global expert` 在少数异常点上过度触发："
            f"dominant_global_rate = `{float(router2['dominant_global_rate']):.4f}`；"
            f"最大失败来自 `工况22` 与 `工况30`。"
        ),
        (
            f"- `domain_3` 的退化更像“baseline 已高估，global residual 继续顺着同方向加码”："
            f"`A0` 的 mean_signed_error = `{float(domain3['base_mean_signed_error']):.4f}`，"
            f"`A3` 的 mean_correction_delta = `{float(domain3['mean_correction_delta']):.4f}`，"
            f"router 的 dominant_global_rate = `{float(router3['dominant_global_rate']):.4f}`。"
        ),
        "",
        "## domain 行为表",
        "",
    ]

    for _, row in domain_behavior_df.iterrows():
        lines.append(
            f"- `domain_{int(row['learned_domain_id'])}`: raw_source_mix=`{row['raw_source_mix']}`, "
            f"base_case_mae=`{row['base_case_mae']:.4f}`, moe_case_mae=`{row['moe_case_mae']:.4f}`, "
            f"mean_abs_error_gain=`{row['mean_abs_error_gain']:.4f}`, dominant_expert_mode=`{row['dominant_expert_mode']}`"
        )

    lines.extend(["", "## 关键坏例子", ""])
    for _, row in key_failure_df.head(6).iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}` (`domain_{int(row['learned_domain_id'])}`): "
            f"A0_abs=`{row['abs_error_base']:.4f}`, A3_abs=`{row['abs_error_moe']:.4f}`, "
            f"correction_delta=`{row['correction_delta']:.4f}`, abs_error_gain=`{row['abs_error_gain']:.4f}`"
        )

    lines.extend(["", "## fold 元数据", ""])
    for _, row in fold_df.iterrows():
        lines.append(
            f"- `{row['fold_tag']}`: holdout_case_ids=`{row['holdout_case_ids']}`, raw_source_mix=`{row['raw_source_mix']}`, residual_bound=`{row['residual_bound']:.4f}`"
        )

    lines.extend(["", "## 一句话总结", ""])
    lines.append(
        "截至 `2026-04-09`，这次 `067` full run 更支持把当前 frozen-embedding `A3_sparse_router_moe` 理解为："
        "它能在“高风速混合域的系统性低估”场景里提供方向一致的 prototype correction，"
        "但对 `domain_2` 这类少数异常点和 `domain_3` 这类 baseline 已高估的 final 子域，"
        "当前 global / trigger 行为仍明显不稳。"
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
