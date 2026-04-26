from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_ROOT = REPO_ROOT / "src" / "try" / "066_reuse_embedding_domain_split"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from reuse_embedding_domain_common import (  # noqa: E402
    build_cleaned_signal_frames,
    build_embedding_case_table,
    build_record_table,
    get_embedding_columns,
    load_fixed_window_embeddings,
    load_source_catalog,
    load_try053_module,
)

TRY_NAME = "069_added2_embedding_pca_projection"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
DOMAIN_COLORS = {
    "final": "#4c78a8",
    "added": "#f58518",
    "added2": "#e45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用 057 的 2s+8s embedding，为 added2 输出 PCA 二维投影图。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try053 = load_try053_module()
    catalog = load_source_catalog()
    record_df = build_record_table(catalog)
    _, cleaned_signal_frames = build_cleaned_signal_frames(catalog.all_records)
    fixed_window_embeddings = load_fixed_window_embeddings(
        try053=try053,
        export_records=catalog.all_records,
        cleaned_signal_frames=cleaned_signal_frames,
    )
    embedding_case_df = build_embedding_case_table(record_df, fixed_window_embeddings)
    embedding_columns = get_embedding_columns(embedding_case_df)
    pca_df, pca = build_pca_projection(embedding_case_df, embedding_columns)

    added2_summary_df = (
        pca_df.loc[pca_df["raw_source_domain"] == "added2", ["case_id", "file_name", "wind_speed", "rpm", "pca1", "pca2"]]
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    embedding_case_df.to_csv(output_dir / "embedding_case_table.csv", index=False, encoding="utf-8-sig")
    pca_df.to_csv(output_dir / "embedding_pca_coords.csv", index=False, encoding="utf-8-sig")
    added2_summary_df.to_csv(output_dir / "added2_projection_summary.csv", index=False, encoding="utf-8-sig")

    create_pca_by_source_plot(pca_df, plot_dir / "pca_by_source_domain.png")
    create_added2_focus_plot(pca_df, plot_dir / "pca_added2_focus.png")
    create_projection_panel_plot(pca_df, plot_dir / "pca_added2_projection_panel.png")
    write_summary_markdown(output_dir / "summary.md", pca_df=pca_df, pca=pca)

    print("069 added2 embedding PCA projection 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")


def build_pca_projection(
    embedding_case_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, PCA]:
    matrix = embedding_case_df[embedding_columns].to_numpy(dtype=float, copy=False)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    pca_df = embedding_case_df[
        ["case_id", "file_name", "display_name", "raw_source_domain", "wind_speed", "rpm", "is_labeled"]
    ].copy()
    pca_df["pca1"] = coords[:, 0]
    pca_df["pca2"] = coords[:, 1]
    return pca_df, pca


def create_pca_by_source_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for domain_name in ("final", "added", "added2"):
        block = pca_df.loc[pca_df["raw_source_domain"] == domain_name]
        if block.empty:
            continue
        ax.scatter(
            block["pca1"],
            block["pca2"],
            s=95 if domain_name != "added2" else 130,
            alpha=0.86,
            label=domain_name,
            color=DOMAIN_COLORS[domain_name],
            edgecolors="black" if domain_name == "added2" else "none",
            linewidths=0.8 if domain_name == "added2" else 0.0,
        )
    for _, row in pca_df.iterrows():
        ax.text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=8, ha="center", va="center")
    ax.set_title("TinyTCN 2s+8s embedding PCA by source domain")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_added2_focus_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    background = pca_df.loc[pca_df["raw_source_domain"] != "added2"]
    added2 = pca_df.loc[pca_df["raw_source_domain"] == "added2"]

    ax.scatter(
        background["pca1"],
        background["pca2"],
        s=65,
        alpha=0.18,
        color="#9aa0a6",
        label="final + added background",
    )
    ax.scatter(
        added2["pca1"],
        added2["pca2"],
        s=170,
        alpha=0.95,
        color=DOMAIN_COLORS["added2"],
        edgecolors="black",
        linewidths=1.1,
        label="added2",
    )
    for _, row in added2.iterrows():
        ax.text(
            row["pca1"],
            row["pca2"],
            f"{int(row['case_id'])}",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    ax.set_title("Added2 projection in TinyTCN 2s+8s embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_projection_panel_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for domain_name in ("final", "added", "added2"):
        block = pca_df.loc[pca_df["raw_source_domain"] == domain_name]
        axes[0].scatter(
            block["pca1"],
            block["pca2"],
            s=90 if domain_name != "added2" else 120,
            alpha=0.85,
            label=domain_name,
            color=DOMAIN_COLORS[domain_name],
            edgecolors="black" if domain_name == "added2" else "none",
            linewidths=0.8 if domain_name == "added2" else 0.0,
        )
    for _, row in pca_df.iterrows():
        axes[0].text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=8, ha="center", va="center")
    axes[0].set_title("All source domains")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    background = pca_df.loc[pca_df["raw_source_domain"] != "added2"]
    added2 = pca_df.loc[pca_df["raw_source_domain"] == "added2"]
    axes[1].scatter(background["pca1"], background["pca2"], s=60, alpha=0.18, color="#9aa0a6", label="background")
    axes[1].scatter(
        added2["pca1"],
        added2["pca2"],
        s=170,
        alpha=0.95,
        color=DOMAIN_COLORS["added2"],
        edgecolors="black",
        linewidths=1.1,
        label="added2",
    )
    for _, row in added2.iterrows():
        axes[1].text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=10, fontweight="bold", ha="left", va="bottom")
    axes[1].set_title("Added2 focus")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    fig.suptitle("TinyTCN 2s+8s embedding PCA projection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(output_path: Path, *, pca_df: pd.DataFrame, pca: PCA) -> None:
    lines = [
        "# added2 embedding PCA projection",
        "",
        "## 摘要",
        "",
        f"- 导出工况数：`{len(pca_df)}`",
        f"- `final` 工况数：`{int((pca_df['raw_source_domain'] == 'final').sum())}`",
        f"- `added` 工况数：`{int((pca_df['raw_source_domain'] == 'added').sum())}`",
        f"- `added2` 工况数：`{int((pca_df['raw_source_domain'] == 'added2').sum())}`",
        f"- PCA explained variance：`PC1={pca.explained_variance_ratio_[0]:.2%}`, `PC2={pca.explained_variance_ratio_[1]:.2%}`",
        "",
        "## added2 投影坐标",
        "",
    ]
    added2_df = pca_df.loc[pca_df["raw_source_domain"] == "added2"].sort_values("case_id")
    for _, row in added2_df.iterrows():
        wind_text = "nan" if pd.isna(row["wind_speed"]) else f"{float(row['wind_speed']):.4f}"
        rpm_text = "nan" if pd.isna(row["rpm"]) else f"{float(row['rpm']):.4f}"
        lines.append(
            f"- `工况{int(row['case_id'])}`: "
            f"`wind={wind_text}`, `rpm={rpm_text}`, "
            f"`pca1={float(row['pca1']):.4f}`, `pca2={float(row['pca2']):.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
