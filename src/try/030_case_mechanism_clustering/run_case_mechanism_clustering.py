from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT,
    REPO_ROOT / "src" / "try" / "009_phase1_feature_groups",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import (
    build_dataset_inventory,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig, build_case_feature_frame

TRY_NAME = "030_case_mechanism_clustering"
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
QUALITY_PATH = REPO_ROOT / "outputs" / "data_quality_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="做工况级机制特征聚类与可视化。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }

    mechanism_df = build_case_mechanism_table(records, cleaned_signal_frames, common_signal_columns)
    feature_columns = get_mechanism_feature_columns()
    embedding_df, cluster_summary_df, pca_model, scaler, ordered_cases = cluster_cases(mechanism_df, feature_columns)

    mechanism_df.to_csv(output_dir / "case_mechanism_table.csv", index=False, encoding="utf-8-sig")
    embedding_df.to_csv(output_dir / "case_embedding.csv", index=False, encoding="utf-8-sig")
    cluster_summary_df.to_csv(output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    create_pca_scatter(embedding_df, output_dir / "pca_cluster_scatter.png")
    create_heatmap(mechanism_df, feature_columns, ordered_cases, output_dir / "mechanism_heatmap.png")
    write_summary_markdown(
        mechanism_df=mechanism_df,
        embedding_df=embedding_df,
        cluster_summary_df=cluster_summary_df,
        pca_model=pca_model,
        output_path=output_dir / "summary.md",
    )

    print("工况机制聚类探索已完成。")
    print(f"输出目录: {output_dir}")


def build_case_mechanism_table(
    records: list,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    common_signal_columns: list[str],
) -> pd.DataFrame:
    quality_df = pd.read_csv(QUALITY_PATH)
    inventory_df = build_dataset_inventory(records)
    wind_case_df = pd.read_csv(WIND_CASE_PATH)
    rpm_case_df = pd.read_csv(RPM_CASE_PATH)

    strain_channels = [col for col in common_signal_columns if "应变" in col]
    acc_channels = [col for col in common_signal_columns if "Acc" in col]

    rows: list[dict[str, object]] = []
    for record in records:
        signal_df = cleaned_signal_frames[record.case_id]
        feature_df = build_case_feature_frame(record, signal_df, WindowConfig())

        per_case = {
            "case_id": record.case_id,
            "file_name": record.file_name,
            "display_name": record.display_name,
            "wind_speed": record.wind_speed,
            "rpm": record.rpm,
        }
        per_case.update(
            summarize_case_windows(feature_df, strain_channels=strain_channels, acc_channels=acc_channels)
        )
        rows.append(per_case)

    mechanism_df = pd.DataFrame(rows)
    mechanism_df = mechanism_df.merge(
        quality_df[
            [
                "case_id",
                "missing_ratio_in_common_cols",
                "edge_removed_ratio",
                "rows_after_edge_drop",
                "windows_total",
            ]
        ],
        on="case_id",
        how="left",
    )
    mechanism_df = mechanism_df.merge(
        inventory_df[["case_id", "duration_seconds", "sampling_hz_est"]],
        on="case_id",
        how="left",
    )

    wind_best = wind_case_df.loc[
        wind_case_df.groupby("case_id")["abs_error"].idxmin(),
        ["case_id", "window_label", "abs_error"],
    ].rename(columns={"window_label": "wind_best_window", "abs_error": "wind_best_window_error"})
    wind_5s = wind_case_df[wind_case_df["window_label"] == "5s"][
        ["case_id", "abs_error", "pred_mean"]
    ].rename(columns={"abs_error": "wind_loco_error_5s", "pred_mean": "wind_pred_5s"})
    rpm_best = rpm_case_df.loc[
        rpm_case_df.groupby("case_id")["abs_error"].idxmin(),
        ["case_id", "window_label", "abs_error"],
    ].rename(columns={"window_label": "rpm_best_window", "abs_error": "rpm_best_window_error"})

    mechanism_df = mechanism_df.merge(wind_best, on="case_id", how="left")
    mechanism_df = mechanism_df.merge(wind_5s, on="case_id", how="left")
    mechanism_df = mechanism_df.merge(rpm_best, on="case_id", how="left")
    return mechanism_df.sort_values("case_id").reset_index(drop=True)


def summarize_case_windows(
    feature_df: pd.DataFrame,
    strain_channels: list[str],
    acc_channels: list[str],
) -> dict[str, float]:
    def channel_metric_columns(channels: list[str], suffix: str) -> list[str]:
        return [f"{channel}__{suffix}" for channel in channels]

    def per_window_mean(columns: list[str]) -> pd.Series:
        return feature_df[columns].mean(axis=1)

    def median_iqr(series: pd.Series, prefix: str) -> dict[str, float]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        return {
            f"{prefix}_median": float(series.median()),
            f"{prefix}_iqr": q3 - q1,
        }

    strain_rms = per_window_mean(channel_metric_columns(strain_channels, "rms"))
    acc_rms = per_window_mean(channel_metric_columns(acc_channels, "rms"))
    strain_std = per_window_mean(channel_metric_columns(strain_channels, "std"))
    acc_std = per_window_mean(channel_metric_columns(acc_channels, "std"))
    strain_peak = per_window_mean(channel_metric_columns(strain_channels, "fft_peak_freq"))
    acc_peak = per_window_mean(channel_metric_columns(acc_channels, "fft_peak_freq"))
    strain_energy = per_window_mean(channel_metric_columns(strain_channels, "fft_total_energy"))
    acc_energy = per_window_mean(channel_metric_columns(acc_channels, "fft_total_energy"))
    strain_low = per_window_mean(channel_metric_columns(strain_channels, "fft_band_ratio_0_2hz"))
    strain_mid = per_window_mean(channel_metric_columns(strain_channels, "fft_band_ratio_2_5hz"))
    strain_high = per_window_mean(channel_metric_columns(strain_channels, "fft_band_ratio_5_10hz"))
    acc_low = per_window_mean(channel_metric_columns(acc_channels, "fft_band_ratio_0_2hz"))
    acc_mid = per_window_mean(channel_metric_columns(acc_channels, "fft_band_ratio_2_5hz"))
    acc_high = per_window_mean(channel_metric_columns(acc_channels, "fft_band_ratio_5_10hz"))

    result: dict[str, float] = {}
    for series, prefix in (
        (strain_rms, "strain_rms"),
        (acc_rms, "acc_rms"),
        (strain_std, "strain_std"),
        (acc_std, "acc_std"),
        (strain_peak, "strain_peak_freq"),
        (acc_peak, "acc_peak_freq"),
        (strain_energy, "strain_energy"),
        (acc_energy, "acc_energy"),
        (strain_low, "strain_low_ratio"),
        (strain_mid, "strain_mid_ratio"),
        (strain_high, "strain_high_ratio"),
        (acc_low, "acc_low_ratio"),
        (acc_mid, "acc_mid_ratio"),
        (acc_high, "acc_high_ratio"),
    ):
        result.update(median_iqr(series, prefix))

    result["window_missing_ratio_mean"] = float(feature_df["raw_missing_ratio"].mean())
    result["window_missing_ratio_max"] = float(feature_df["raw_missing_ratio"].max())
    result["strain_acc_rms_ratio"] = float(result["strain_rms_median"] / max(result["acc_rms_median"], 1e-12))
    result["strain_acc_energy_ratio"] = float(result["strain_energy_median"] / max(result["acc_energy_median"], 1e-12))
    return result


def get_mechanism_feature_columns() -> list[str]:
    return [
        "missing_ratio_in_common_cols",
        "edge_removed_ratio",
        "rows_after_edge_drop",
        "windows_total",
        "duration_seconds",
        "window_missing_ratio_mean",
        "window_missing_ratio_max",
        "strain_rms_median",
        "strain_rms_iqr",
        "acc_rms_median",
        "acc_rms_iqr",
        "strain_std_median",
        "acc_std_median",
        "strain_peak_freq_median",
        "acc_peak_freq_median",
        "strain_energy_median",
        "acc_energy_median",
        "strain_low_ratio_median",
        "strain_mid_ratio_median",
        "strain_high_ratio_median",
        "acc_low_ratio_median",
        "acc_mid_ratio_median",
        "acc_high_ratio_median",
        "strain_acc_rms_ratio",
        "strain_acc_energy_ratio",
    ]


def cluster_cases(
    mechanism_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, PCA, StandardScaler, list[int]]:
    X = mechanism_df[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    embedding_df = mechanism_df[
        [
            "case_id",
            "file_name",
            "wind_speed",
            "rpm",
            "wind_loco_error_5s",
            "wind_best_window",
            "rpm_best_window",
        ]
    ].copy()
    embedding_df["cluster_id"] = best_labels
    embedding_df["pca1"] = coords[:, 0]
    embedding_df["pca2"] = coords[:, 1]
    embedding_df["cluster_count"] = best_k
    embedding_df["silhouette_score"] = best_score

    ordered_cases = (
        embedding_df.sort_values(["cluster_id", "pca1", "case_id"])["case_id"].tolist()
    )
    cluster_summary_df = build_cluster_summary(mechanism_df, feature_columns, embedding_df)
    return embedding_df, cluster_summary_df, pca, scaler, ordered_cases


def build_cluster_summary(
    mechanism_df: pd.DataFrame,
    feature_columns: list[str],
    embedding_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = mechanism_df.merge(
        embedding_df[["case_id", "cluster_id"]],
        on="case_id",
        how="left",
    )
    z_df = merged.copy()
    z_df[feature_columns] = StandardScaler().fit_transform(z_df[feature_columns].to_numpy(dtype=float))

    rows: list[dict[str, object]] = []
    for cluster_id, block in z_df.groupby("cluster_id", sort=True):
        feature_means = block[feature_columns].mean().sort_values(ascending=False)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "case_ids": ",".join(str(int(v)) for v in block["case_id"].sort_values().tolist()),
                "case_count": int(len(block)),
                "wind_loco_error_5s_mean": float(block["wind_loco_error_5s"].mean()),
                "top_positive_feature_1": feature_means.index[0],
                "top_positive_feature_1_z": float(feature_means.iloc[0]),
                "top_positive_feature_2": feature_means.index[1],
                "top_positive_feature_2_z": float(feature_means.iloc[1]),
                "top_negative_feature_1": feature_means.index[-1],
                "top_negative_feature_1_z": float(feature_means.iloc[-1]),
                "top_negative_feature_2": feature_means.index[-2],
                "top_negative_feature_2_z": float(feature_means.iloc[-2]),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def create_pca_scatter(embedding_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    clusters = sorted(embedding_df["cluster_id"].unique())
    cmap = plt.get_cmap("tab10")

    for cluster_id in clusters:
        block = embedding_df[embedding_df["cluster_id"] == cluster_id]
        axes[0].scatter(block["pca1"], block["pca2"], s=90, label=f"cluster {cluster_id}", color=cmap(cluster_id))
    for _, row in embedding_df.iterrows():
        axes[0].annotate(str(int(row["case_id"])), (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    axes[0].set_title("PCA colored by cluster")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    scatter = axes[1].scatter(
        embedding_df["pca1"],
        embedding_df["pca2"],
        c=embedding_df["wind_loco_error_5s"],
        s=110,
        cmap="viridis",
    )
    for _, row in embedding_df.iterrows():
        label = f"{int(row['case_id'])}|{row['wind_best_window']}"
        axes[1].annotate(label, (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_title("PCA colored by TinyTCN@5s LOCO error")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[1], label="wind_loco_error_5s")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_heatmap(
    mechanism_df: pd.DataFrame,
    feature_columns: list[str],
    ordered_cases: list[int],
    output_path: Path,
) -> None:
    ordered_df = mechanism_df.set_index("case_id").loc[ordered_cases].reset_index()
    X = ordered_df[feature_columns].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(X_scaled, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(ordered_df)))
    ax.set_yticklabels([str(int(v)) for v in ordered_df["case_id"]])
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=75, ha="right", fontsize=8)
    ax.set_title("Per-case mechanism features (z-scored)")
    ax.set_xlabel("feature")
    ax.set_ylabel("case_id (sorted by cluster, PC1)")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    mechanism_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    pca_model: PCA,
    output_path: Path,
) -> None:
    lines = [
        "# 工况机制聚类探索结论",
        "",
        f"- 工况数：`{len(mechanism_df)}`",
        f"- 选定 cluster 数：`{int(embedding_df['cluster_count'].iloc[0])}`",
        f"- silhouette score：`{float(embedding_df['silhouette_score'].iloc[0]):.4f}`",
        f"- PCA explained variance：`PC1={pca_model.explained_variance_ratio_[0]:.2%}`, `PC2={pca_model.explained_variance_ratio_[1]:.2%}`",
        "",
        "## 聚类结果",
        "",
    ]
    for _, row in cluster_summary_df.iterrows():
        lines.append(
            f"- `cluster {int(row['cluster_id'])}`: case_ids=`{row['case_ids']}`, wind_loco_error_5s_mean=`{row['wind_loco_error_5s_mean']:.4f}`, top+=`{row['top_positive_feature_1']}`, `{row['top_positive_feature_2']}`, top-=`{row['top_negative_feature_1']}`, `{row['top_negative_feature_2']}`"
        )

    hard_cases = mechanism_df.nlargest(4, "wind_loco_error_5s")[["case_id", "wind_loco_error_5s"]]
    lines.extend(["", "## 难工况位置", ""])
    for _, row in hard_cases.iterrows():
        cluster_id = int(embedding_df.loc[embedding_df["case_id"] == row["case_id"], "cluster_id"].iloc[0])
        best_window = embedding_df.loc[embedding_df["case_id"] == row["case_id"], "wind_best_window"].iloc[0]
        rpm_best = embedding_df.loc[embedding_df["case_id"] == row["case_id"], "rpm_best_window"].iloc[0]
        lines.append(
            f"- `工况{int(row['case_id'])}`: cluster=`{cluster_id}`, wind_loco_error_5s=`{row['wind_loco_error_5s']:.4f}`, wind_best_window=`{best_window}`, rpm_best_window=`{rpm_best}`"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 聚类只使用机制特征，不把风速、RPM、误差或最优窗长直接喂进聚类。",
            "- 风速误差与窗长偏好只作为结果注记，用来帮助解释聚类。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
