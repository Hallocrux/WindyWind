from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "031_case_error_mode_clustering"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME

WIND_CASE_PATH = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "014_phase3_tcn_window_length_scan"
    / "tcn_window_scan_case_level_predictions.csv"
)
RPM_CASE_PATH = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "024_tinytcn_rpm_fine_window_scan"
    / "rpm_fine_window_scan_case_level_predictions.csv"
)
BOUNDARY_PATH = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "003_start_end_segment_diagnosis"
    / "segment_distance_summary.csv"
)
MECHANISM_EMBED_PATH = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "030_case_mechanism_clustering"
    / "case_embedding.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="做工况级误差模式聚类。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_df = build_error_profile_table()
    feature_columns = get_error_feature_columns()
    embedding_df, cluster_summary_df, pca_model, ordered_cases = cluster_profiles(profile_df, feature_columns)

    profile_df.to_csv(output_dir / "case_error_profile_table.csv", index=False, encoding="utf-8-sig")
    embedding_df.to_csv(output_dir / "case_error_embedding.csv", index=False, encoding="utf-8-sig")
    cluster_summary_df.to_csv(output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    create_pca_scatter(embedding_df, output_dir / "error_mode_pca_scatter.png")
    create_heatmap(profile_df, feature_columns, ordered_cases, output_dir / "error_mode_heatmap.png")
    write_summary_markdown(profile_df, embedding_df, cluster_summary_df, pca_model, output_dir / "summary.md")

    print("工况误差模式聚类探索已完成。")
    print(f"输出目录: {output_dir}")


def build_error_profile_table() -> pd.DataFrame:
    wind_df = pd.read_csv(WIND_CASE_PATH)
    rpm_df = pd.read_csv(RPM_CASE_PATH)
    boundary_df = pd.read_csv(BOUNDARY_PATH)
    mechanism_df = pd.read_csv(MECHANISM_EMBED_PATH)

    wind_pivot = wind_df.pivot_table(index=["case_id", "file_name", "true_wind_speed"], columns="window_label", values="abs_error").reset_index()
    wind_pivot = wind_pivot.rename(columns={"2s": "wind_err_2s", "4s": "wind_err_4s", "5s": "wind_err_5s", "8s": "wind_err_8s"})
    wind_pivot["wind_best_err"] = wind_pivot[["wind_err_2s", "wind_err_4s", "wind_err_5s", "wind_err_8s"]].min(axis=1)
    wind_pivot["wind_long_short_delta"] = wind_pivot["wind_err_8s"] - wind_pivot["wind_err_2s"]

    rpm_pivot = rpm_df.pivot_table(index=["case_id", "file_name", "true_rpm"], columns="window_label", values="abs_error").reset_index()
    rpm_pivot = rpm_pivot.rename(
        columns={
            "2.0s": "rpm_err_2_0s",
            "2.5s": "rpm_err_2_5s",
            "3.0s": "rpm_err_3_0s",
            "3.5s": "rpm_err_3_5s",
            "4.0s": "rpm_err_4_0s",
            "4.5s": "rpm_err_4_5s",
            "5.0s": "rpm_err_5_0s",
        }
    )
    rpm_error_cols = [col for col in rpm_pivot.columns if col.startswith("rpm_err_")]
    rpm_pivot["rpm_best_err"] = rpm_pivot[rpm_error_cols].min(axis=1)
    rpm_pivot["rpm_5s_minus_3s"] = rpm_pivot["rpm_err_5_0s"] - rpm_pivot["rpm_err_3_0s"]

    merged = wind_pivot.merge(rpm_pivot, on="case_id", how="inner")
    merged = merged.merge(
        boundary_df[
            [
                "case_id",
                "start_middle_vs_within",
                "end_middle_vs_within",
                "start_end_vs_within",
            ]
        ],
        on="case_id",
        how="left",
    )
    merged = merged.merge(
        mechanism_df[
            [
                "case_id",
                "cluster_id",
                "wind_best_window",
                "rpm_best_window",
                "wind_loco_error_5s",
            ]
        ],
        on="case_id",
        how="left",
        suffixes=("", "_mech"),
    )
    boundary_cols = ["start_middle_vs_within", "end_middle_vs_within", "start_end_vs_within"]
    merged["boundary_metrics_missing"] = merged[boundary_cols].isna().any(axis=1).astype(int)
    for col in boundary_cols:
        merged[col] = merged[col].fillna(float(merged[col].median()))
    return merged.sort_values("case_id").reset_index(drop=True)


def get_error_feature_columns() -> list[str]:
    return [
        "wind_err_2s",
        "wind_err_4s",
        "wind_err_5s",
        "wind_err_8s",
        "wind_best_err",
        "wind_long_short_delta",
        "rpm_err_2_0s",
        "rpm_err_2_5s",
        "rpm_err_3_0s",
        "rpm_err_3_5s",
        "rpm_err_4_0s",
        "rpm_err_4_5s",
        "rpm_err_5_0s",
        "rpm_best_err",
        "rpm_5s_minus_3s",
        "start_middle_vs_within",
        "end_middle_vs_within",
        "start_end_vs_within",
        "boundary_metrics_missing",
    ]


def cluster_profiles(
    profile_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, PCA, list[int]]:
    X = profile_df[feature_columns].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    best_score = -np.inf
    best_labels = None
    best_k = None
    for k in (2, 3, 4):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    embedding_df = profile_df[
        [
            "case_id",
            "file_name_x",
            "true_wind_speed",
            "true_rpm",
            "wind_loco_error_5s",
            "cluster_id",
            "wind_best_window",
            "rpm_best_window",
        ]
    ].copy()
    embedding_df = embedding_df.rename(columns={"file_name_x": "file_name", "cluster_id": "mechanism_cluster_id"})
    embedding_df["error_cluster_id"] = best_labels
    embedding_df["pca1"] = coords[:, 0]
    embedding_df["pca2"] = coords[:, 1]
    embedding_df["cluster_count"] = best_k
    embedding_df["silhouette_score"] = best_score

    ordered_cases = embedding_df.sort_values(["error_cluster_id", "pca1", "case_id"])["case_id"].tolist()
    cluster_summary_df = build_cluster_summary(profile_df, feature_columns, embedding_df)
    return embedding_df, cluster_summary_df, pca, ordered_cases


def build_cluster_summary(profile_df: pd.DataFrame, feature_columns: list[str], embedding_df: pd.DataFrame) -> pd.DataFrame:
    merged = profile_df.merge(
        embedding_df[["case_id", "error_cluster_id"]],
        on="case_id",
        how="left",
    )
    z_df = merged.copy()
    z_df[feature_columns] = StandardScaler().fit_transform(z_df[feature_columns].to_numpy(dtype=float))
    rows: list[dict[str, object]] = []
    for cluster_id, block in z_df.groupby("error_cluster_id", sort=True):
        feat_means = block[feature_columns].mean().sort_values(ascending=False)
        rows.append(
            {
                "error_cluster_id": int(cluster_id),
                "case_ids": ",".join(str(int(v)) for v in block["case_id"].sort_values().tolist()),
                "case_count": int(len(block)),
                "mechanism_clusters": ",".join(str(int(v)) for v in sorted(block["cluster_id"].unique())),
                "wind_loco_error_5s_mean": float(block["wind_loco_error_5s"].mean()),
                "top_positive_feature_1": feat_means.index[0],
                "top_positive_feature_1_z": float(feat_means.iloc[0]),
                "top_positive_feature_2": feat_means.index[1],
                "top_positive_feature_2_z": float(feat_means.iloc[1]),
                "top_negative_feature_1": feat_means.index[-1],
                "top_negative_feature_1_z": float(feat_means.iloc[-1]),
                "top_negative_feature_2": feat_means.index[-2],
                "top_negative_feature_2_z": float(feat_means.iloc[-2]),
            }
        )
    return pd.DataFrame(rows).sort_values("error_cluster_id").reset_index(drop=True)


def create_pca_scatter(embedding_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")
    for cluster_id in sorted(embedding_df["error_cluster_id"].unique()):
        block = embedding_df[embedding_df["error_cluster_id"] == cluster_id]
        axes[0].scatter(block["pca1"], block["pca2"], s=90, label=f"error cluster {cluster_id}", color=cmap(cluster_id))
    for _, row in embedding_df.iterrows():
        axes[0].annotate(str(int(row["case_id"])), (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    axes[0].set_title("Error-profile PCA colored by error cluster")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    scatter = axes[1].scatter(
        embedding_df["pca1"],
        embedding_df["pca2"],
        c=embedding_df["mechanism_cluster_id"],
        s=110,
        cmap="coolwarm",
    )
    for _, row in embedding_df.iterrows():
        label = f"{int(row['case_id'])}|m{int(row['mechanism_cluster_id'])}"
        axes[1].annotate(label, (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_title("Error-profile PCA colored by mechanism cluster")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[1], label="mechanism_cluster_id")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_heatmap(profile_df: pd.DataFrame, feature_columns: list[str], ordered_cases: list[int], output_path: Path) -> None:
    ordered_df = profile_df.set_index("case_id").loc[ordered_cases].reset_index()
    X = ordered_df[feature_columns].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(X_scaled, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(ordered_df)))
    ax.set_yticklabels([str(int(v)) for v in ordered_df["case_id"]])
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=75, ha="right", fontsize=8)
    ax.set_title("Per-case error profile features (z-scored)")
    ax.set_xlabel("feature")
    ax.set_ylabel("case_id (sorted by error cluster, PC1)")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    profile_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_model: PCA,
    output_path: Path,
) -> None:
    lines = [
        "# 工况误差模式聚类探索结论",
        "",
        f"- 工况数：`{len(profile_df)}`",
        f"- 选定 error cluster 数：`{int(embedding_df['cluster_count'].iloc[0])}`",
        f"- silhouette score：`{float(embedding_df['silhouette_score'].iloc[0]):.4f}`",
        f"- PCA explained variance：`PC1={pca_model.explained_variance_ratio_[0]:.2%}`, `PC2={pca_model.explained_variance_ratio_[1]:.2%}`",
        "",
        "## 聚类结果",
        "",
    ]
    for _, row in cluster_summary_df.iterrows():
        lines.append(
            f"- `error cluster {int(row['error_cluster_id'])}`: case_ids=`{row['case_ids']}`, mechanism_clusters=`{row['mechanism_clusters']}`, wind_loco_error_5s_mean=`{row['wind_loco_error_5s_mean']:.4f}`, top+=`{row['top_positive_feature_1']}`, `{row['top_positive_feature_2']}`, top-=`{row['top_negative_feature_1']}`, `{row['top_negative_feature_2']}`"
        )

    hard_cases = embedding_df.nlargest(4, "wind_loco_error_5s")[["case_id", "wind_loco_error_5s", "error_cluster_id", "mechanism_cluster_id", "wind_best_window", "rpm_best_window"]]
    lines.extend(["", "## 难工况位置", ""])
    for _, row in hard_cases.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: error_cluster=`{int(row['error_cluster_id'])}`, mechanism_cluster=`{int(row['mechanism_cluster_id'])}`, wind_loco_error_5s=`{row['wind_loco_error_5s']:.4f}`, wind_best_window=`{row['wind_best_window']}`, rpm_best_window=`{row['rpm_best_window']}`"
        )

    lines.extend([
        "",
        "## 说明",
        "",
        "- 误差聚类只使用误差画像与边界差异特征，不直接使用原始机制特征。",
        "- 机制簇只作为对照标签，用来观察两层结构是否一致。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
