from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import run_contrastive_condition_embedding as base


OUTPUT_DIR = base.OUTPUT_DIR
DOMAIN_COLORS = {
    "final": "#4c78a8",
    "added": "#f58518",
    "added2": "#e45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 086 contrastive encoder 导出全部工况 embedding 并做 PCA 二维投影。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "models" / "checkpoints"

    try053 = base.load_try053_module()
    catalog = base.load_source_catalog()
    all_records = catalog.all_records
    train_records = sorted(
        [record for record in catalog.final_records if record.is_labeled] + catalog.added_records,
        key=lambda record: record.case_id,
    )
    record_df = base.build_record_table(catalog)
    _, cleaned_signal_frames = base.build_cleaned_signal_frames(all_records)

    per_window: dict[str, dict[str, object]] = {}
    config = base.TrainConfig()
    for window_label in base.WINDOW_LABELS:
        per_window[window_label] = base.load_or_train_window_embeddings(
            try053=try053,
            window_label=window_label,
            train_records=train_records,
            export_records=all_records,
            cleaned_signal_frames=cleaned_signal_frames,
            config=config,
            seed=42,
            ckpt_dir=ckpt_dir,
            force_retrain=False,
        )

    embedding_case_df = build_all_embedding_case_table(record_df, per_window)
    embedding_columns = [column for column in embedding_case_df.columns if column.startswith("embedding_")]
    pca_df, pca = build_pca_projection(embedding_case_df, embedding_columns)

    embedding_case_df.to_csv(output_dir / "all_case_embedding_table.csv", index=False, encoding="utf-8-sig")
    pca_df.to_csv(output_dir / "all_case_embedding_pca_coords.csv", index=False, encoding="utf-8-sig")
    create_pca_plot(pca_df, plot_dir / "all_case_embedding_pca.png")
    write_summary(output_dir / "all_case_embedding_pca_summary.md", pca_df, pca, embedding_columns)

    print("086 all-case contrastive embedding PCA projection finished.")
    print(f"output_dir={output_dir}")
    print(f"case_count={len(pca_df)}")
    print(f"embedding_dim={len(embedding_columns)}")
    print(f"pca_explained_variance=({pca.explained_variance_ratio_[0]:.4f}, {pca.explained_variance_ratio_[1]:.4f})")


def build_all_embedding_case_table(record_df: pd.DataFrame, per_window: dict[str, dict[str, object]]) -> pd.DataFrame:
    result = record_df.copy().sort_values("case_id").reset_index(drop=True)
    for window_label in base.WINDOW_LABELS:
        embedding_df = base.aggregate_case_embeddings(
            per_window[window_label]["export_meta_df"],
            per_window[window_label]["export_embedding"],
            f"embedding_{window_label}",
        )
        result = result.merge(embedding_df, on="case_id", how="left")

    result["embedding_concat"] = result.apply(
        lambda row: np.concatenate(
            [
                np.asarray(row["embedding_2s"], dtype=float),
                np.asarray(row["embedding_8s"], dtype=float),
            ]
        ).astype(float),
        axis=1,
    )
    matrix = np.vstack(result["embedding_concat"].to_numpy())
    embedding_df = pd.DataFrame(
        matrix,
        columns=[f"embedding_{index + 1}" for index in range(matrix.shape[1])],
        index=result.index,
    )
    return pd.concat(
        [result.drop(columns=["embedding_2s", "embedding_8s", "embedding_concat"]), embedding_df],
        axis=1,
    ).reset_index(drop=True)


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
    return pca_df, pca


def create_pca_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for domain_name, block in pca_df.groupby("raw_source_domain", sort=False):
        ax.scatter(
            block["pca1"],
            block["pca2"],
            s=95 if domain_name != "added2" else 130,
            alpha=0.88,
            label=domain_name,
            color=DOMAIN_COLORS.get(domain_name, "#777777"),
            edgecolors="black" if domain_name in {"added", "added2"} else "none",
            linewidths=0.8 if domain_name in {"added", "added2"} else 0.0,
        )
    for _, row in pca_df.iterrows():
        ax.text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=8, ha="center", va="center")
    ax.set_title("086 contrastive condition embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(output_path: Path, pca_df: pd.DataFrame, pca: PCA, embedding_columns: list[str]) -> None:
    lines = [
        "# 086 all-case contrastive embedding PCA",
        "",
        "- 状态：`current`",
        "- 首次确认：`2026-04-12`",
        "- 最近复核：`2026-04-12`",
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
