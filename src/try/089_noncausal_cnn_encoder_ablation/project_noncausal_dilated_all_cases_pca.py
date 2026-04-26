from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "089_noncausal_cnn_encoder_ablation"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_EMBEDDING_TABLE_PATH = OUTPUT_ROOT / "noncausal_dilated" / "embedding_case_table.csv"

DOMAIN_STYLES = {
    "final": {"color": "#4c78a8", "marker": "o", "size": 90},
    "added": {"color": "#f58518", "marker": "^", "size": 115},
    "added2": {"color": "#e45756", "marker": "s", "size": 125},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project all noncausal_dilated case embeddings to 2D PCA.")
    parser.add_argument("--embedding-table", type=Path, default=DEFAULT_EMBEDDING_TABLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "noncausal_dilated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    embedding_case_df = pd.read_csv(args.embedding_table, encoding="utf-8-sig")
    embedding_columns = [column for column in embedding_case_df.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError(f"没有找到 embedding 列: {args.embedding_table}")

    pca_df, pca = build_pca_projection(embedding_case_df, embedding_columns)
    pca_df.to_csv(output_dir / "all_case_embedding_pca_coords.csv", index=False, encoding="utf-8-sig")
    create_pca_plot(pca_df, plot_dir / "all_case_embedding_pca.png", pca)
    create_pca_plot(
        pca_df.loc[pca_df["raw_source_domain"].isin(["added", "added2"])].copy(),
        plot_dir / "external_case_embedding_pca.png",
        pca,
    )
    write_summary(output_dir / "all_case_embedding_pca_summary.md", pca_df, pca, embedding_columns)

    print("089 noncausal_dilated all-case PCA projection 已完成。")
    print(f"output_dir={output_dir}")
    print(f"case_count={len(pca_df)}")
    print(f"embedding_dim={len(embedding_columns)}")
    print(f"pca_explained_variance=({pca.explained_variance_ratio_[0]:.4f}, {pca.explained_variance_ratio_[1]:.4f})")


def build_pca_projection(embedding_case_df: pd.DataFrame, embedding_columns: list[str]) -> tuple[pd.DataFrame, PCA]:
    matrix = embedding_case_df[embedding_columns].to_numpy(dtype=float, copy=False)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    pca_df = embedding_case_df[
        ["case_id", "file_name", "display_name", "raw_source_domain", "wind_speed", "rpm", "is_labeled"]
    ].copy()
    pca_df["pca1"] = coords[:, 0]
    pca_df["pca2"] = coords[:, 1]
    pca_df["pca_radius"] = np.sqrt(np.square(coords[:, 0]) + np.square(coords[:, 1]))
    return pca_df.sort_values("case_id").reset_index(drop=True), pca


def create_pca_plot(pca_df: pd.DataFrame, output_path: Path, pca: PCA) -> None:
    if pca_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for domain_name, block in pca_df.groupby("raw_source_domain", sort=False):
        style = DOMAIN_STYLES.get(domain_name, {"color": "#777777", "marker": "o", "size": 90})
        ax.scatter(
            block["pca1"],
            block["pca2"],
            s=style["size"],
            marker=style["marker"],
            alpha=0.9,
            label=domain_name,
            color=style["color"],
            edgecolors="black",
            linewidths=0.7,
        )
    for _, row in pca_df.iterrows():
        ax.annotate(
            str(int(row["case_id"])),
            (float(row["pca1"]), float(row["pca2"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("089 noncausal_dilated 2s+8s case embedding PCA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(output_path: Path, pca_df: pd.DataFrame, pca: PCA, embedding_columns: list[str]) -> None:
    lines = [
        "# 089 noncausal_dilated all-case embedding PCA",
        "",
        "- 状态：`current`",
        "- 首次确认：`2026-04-13`",
        "- 最近复核：`2026-04-13`",
        "- 数据范围：`final` 工况 `1-20` + `added` 工况 `21-24` + `added2` 工况 `25-30`",
        "- 代码口径：`src/try/089_noncausal_cnn_encoder_ablation/project_noncausal_dilated_all_cases_pca.py`",
        "- embedding 来源：`outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/embedding_case_table.csv`",
        f"- 导出工况数：`{len(pca_df)}`",
        f"- embedding 维度：`{len(embedding_columns)}`",
        f"- PCA explained variance：`PC1={pca.explained_variance_ratio_[0]:.2%}`, `PC2={pca.explained_variance_ratio_[1]:.2%}`",
        "",
        "## 坐标",
        "",
    ]
    for _, row in pca_df.sort_values("case_id").iterrows():
        wind = "nan" if pd.isna(row["wind_speed"]) else f"{float(row['wind_speed']):.4f}"
        rpm = "nan" if pd.isna(row["rpm"]) else f"{float(row['rpm']):.4f}"
        lines.append(
            f"- `工况{int(row['case_id'])}`: domain=`{row['raw_source_domain']}`, "
            f"wind=`{wind}`, rpm=`{rpm}`, pca1=`{float(row['pca1']):.4f}`, pca2=`{float(row['pca2']):.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
