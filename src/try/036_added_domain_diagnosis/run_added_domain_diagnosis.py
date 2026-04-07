from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY030_ROOT = REPO_ROOT / "src" / "try" / "030_case_mechanism_clustering"
for path in (REPO_ROOT, REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import (
    DatasetRecord,
    build_dataset_inventory,
    get_common_signal_columns,
    prepare_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig, build_case_feature_frame

TRY_NAME = "036_added_domain_diagnosis"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
MECHANISM_EMBEDDING_PATH = REPO_ROOT / "outputs" / "try" / "030_case_mechanism_clustering" / "case_embedding.csv"
ADDED_CLEAN_PRED_PATH = REPO_ROOT / "outputs" / "try" / "034_added_validation_label_check" / "added_case_predictions.csv"
ADDED_FULL_PRED_PATH = REPO_ROOT / "outputs" / "try" / "035_added_validation_with_full_final_pool" / "added_case_predictions.csv"
WINDOW_CONFIG = WindowConfig()
TOP_OUTLIER_COUNT = 8
NEIGHBOR_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 added 工况的域偏移与异常特征。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try030 = _load_try030_module()

    final_records = scan_dataset_records()
    added_records = load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)

    cleaned_signal_frames: dict[int, pd.DataFrame] = {}
    quality_rows: list[dict[str, float | int]] = []
    mechanism_rows: list[dict[str, object]] = []
    window_totals: dict[int, int] = {}

    strain_channels = [column for column in common_signal_columns if "应变" in column]
    acc_channels = [column for column in common_signal_columns if "Acc" in column]

    for record in all_records:
        cleaned, stats = prepare_clean_signal_frame(record, common_signal_columns)
        cleaned_signal_frames[record.case_id] = cleaned
        quality_rows.append(build_quality_row(record, common_signal_columns, stats))
        feature_df = build_case_feature_frame(record, cleaned, WINDOW_CONFIG)
        window_totals[record.case_id] = len(feature_df)
        mechanism_row = {
            "case_id": record.case_id,
            "file_name": record.file_name,
            "display_name": record.display_name,
            "wind_speed": record.wind_speed,
            "rpm": record.rpm,
        }
        mechanism_row.update(
            try030.summarize_case_windows(
                feature_df=feature_df,
                strain_channels=strain_channels,
                acc_channels=acc_channels,
            )
        )
        mechanism_rows.append(mechanism_row)

    quality_df = pd.DataFrame(quality_rows)
    mechanism_df = pd.DataFrame(mechanism_rows)
    inventory_df = pd.concat(
        [build_dataset_inventory(final_records), build_dataset_inventory(added_records)],
        ignore_index=True,
    )
    mechanism_df = mechanism_df.merge(quality_df, on="case_id", how="left")
    mechanism_df = mechanism_df.merge(
        inventory_df[["case_id", "duration_seconds", "sampling_hz_est"]],
        on="case_id",
        how="left",
    )
    mechanism_df["windows_total"] = mechanism_df["case_id"].map(window_totals)
    mechanism_df = mechanism_df.sort_values("case_id").reset_index(drop=True)

    feature_columns = try030.get_mechanism_feature_columns()
    final_mechanism_df = mechanism_df[mechanism_df["case_id"] <= 20].copy().reset_index(drop=True)
    added_mechanism_df = mechanism_df[mechanism_df["case_id"] >= 21].copy().reset_index(drop=True)

    final_cluster_df = pd.read_csv(MECHANISM_EMBEDDING_PATH)[["case_id", "cluster_id"]]
    final_mechanism_df = final_mechanism_df.merge(final_cluster_df, on="case_id", how="left")
    scaler = StandardScaler()
    final_scaled = scaler.fit_transform(final_mechanism_df[feature_columns].to_numpy(dtype=float))
    added_scaled = scaler.transform(added_mechanism_df[feature_columns].to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=42)
    final_coords = pca.fit_transform(final_scaled)
    added_coords = pca.transform(added_scaled)

    nearest_df = build_nearest_neighbor_table(final_mechanism_df, added_mechanism_df, final_scaled, added_scaled)
    outlier_df = build_outlier_table(added_mechanism_df, feature_columns, added_scaled)
    diagnostics_df = build_added_diagnostics(
        added_mechanism_df=added_mechanism_df,
        nearest_df=nearest_df,
        final_mechanism_df=final_mechanism_df,
        final_scaled=final_scaled,
        added_scaled=added_scaled,
        final_coords=final_coords,
        added_coords=added_coords,
    )
    baseline_df = build_baseline_comparison()
    case22_reference_df = build_case22_reference_comparison(
        mechanism_df=mechanism_df,
        nearest_df=nearest_df,
        outlier_df=outlier_df,
    )

    diagnostics_df.to_csv(output_dir / "added_mechanism_diagnostics.csv", index=False, encoding="utf-8-sig")
    nearest_df.to_csv(output_dir / "nearest_final_cases.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(output_dir / "added_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    outlier_df.to_csv(output_dir / "added_feature_outliers.csv", index=False, encoding="utf-8-sig")
    case22_reference_df.to_csv(output_dir / "case22_reference_comparison.csv", index=False, encoding="utf-8-sig")

    create_projection_scatter(
        final_mechanism_df=final_mechanism_df,
        added_mechanism_df=added_mechanism_df,
        final_coords=final_coords,
        added_coords=added_coords,
        output_path=output_dir / "projection_scatter.png",
    )
    create_case22_spectrum_plot(
        records_by_case={record.case_id: record for record in all_records},
        cleaned_signal_frames=cleaned_signal_frames,
        nearest_df=nearest_df,
        common_signal_columns=common_signal_columns,
        output_path=output_dir / "case22_spectrum_comparison.png",
    )
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        common_signal_columns=common_signal_columns,
        final_records=final_records,
        diagnostics_df=diagnostics_df,
        baseline_df=baseline_df,
        outlier_df=outlier_df,
    )

    print("added 域诊断已完成。")
    print(f"输出目录: {output_dir}")


def _load_try030_module():
    module_path = TRY030_ROOT / "run_case_mechanism_clustering.py"
    spec = importlib.util.spec_from_file_location("try030_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_added_records() -> list[DatasetRecord]:
    manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=ADDED_DATA_DIR / f"工况{case_id}.csv",
                wind_speed=float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None,
                rpm=float(row["rpm"]) if not pd.isna(row["rpm"]) else None,
                is_labeled=not pd.isna(row["wind_speed"]) and not pd.isna(row["rpm"]),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return records


def build_quality_row(
    record: DatasetRecord,
    common_signal_columns: list[str],
    stats,
) -> dict[str, float | int]:
    raw_df = pd.read_csv(record.file_path)
    numeric_df = raw_df[common_signal_columns].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(numeric_df.isna().sum().sum())
    return {
        "case_id": record.case_id,
        "rows": int(len(raw_df)),
        "missing_ratio_in_common_cols": float(missing_cells / (len(raw_df) * len(common_signal_columns))),
        "edge_removed_ratio": float(stats.edge_removed_rows / len(raw_df)),
        "rows_after_edge_drop": int(stats.rows_after_edge_drop),
        "rows_after_long_gap_drop": int(stats.rows_after_long_gap_drop),
        "continuous_segment_count": int(stats.continuous_segment_count),
    }


def build_nearest_neighbor_table(
    final_mechanism_df: pd.DataFrame,
    added_mechanism_df: pd.DataFrame,
    final_scaled: np.ndarray,
    added_scaled: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for added_index, added_row in added_mechanism_df.iterrows():
        distances = np.sqrt(np.sum(np.square(final_scaled - added_scaled[added_index]), axis=1))
        order = np.argsort(distances)[:NEIGHBOR_COUNT]
        for rank, final_index in enumerate(order, start=1):
            final_row = final_mechanism_df.iloc[final_index]
            rows.append(
                {
                    "added_case_id": int(added_row["case_id"]),
                    "neighbor_rank": rank,
                    "neighbor_case_id": int(final_row["case_id"]),
                    "neighbor_cluster_id": int(final_row["cluster_id"]),
                    "distance": float(distances[final_index]),
                    "neighbor_wind_speed": float(final_row["wind_speed"]) if pd.notna(final_row["wind_speed"]) else np.nan,
                    "neighbor_rpm": float(final_row["rpm"]) if pd.notna(final_row["rpm"]) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["added_case_id", "neighbor_rank"]).reset_index(drop=True)


def build_outlier_table(
    added_mechanism_df: pd.DataFrame,
    feature_columns: list[str],
    added_scaled: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for added_index, added_row in added_mechanism_df.iterrows():
        z_values = added_scaled[added_index]
        order = np.argsort(np.abs(z_values))[::-1][:TOP_OUTLIER_COUNT]
        for rank, feature_index in enumerate(order, start=1):
            feature_name = feature_columns[feature_index]
            rows.append(
                {
                    "case_id": int(added_row["case_id"]),
                    "outlier_rank": rank,
                    "feature_name": feature_name,
                    "feature_value": float(added_row[feature_name]),
                    "z_score_vs_final": float(z_values[feature_index]),
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "outlier_rank"]).reset_index(drop=True)


def build_added_diagnostics(
    added_mechanism_df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    final_mechanism_df: pd.DataFrame,
    final_scaled: np.ndarray,
    added_scaled: np.ndarray,
    final_coords: np.ndarray,
    added_coords: np.ndarray,
) -> pd.DataFrame:
    cluster_centroids = (
        final_mechanism_df.assign(_row_index=np.arange(len(final_mechanism_df)))
        .groupby("cluster_id")["_row_index"]
        .apply(lambda indexer: final_scaled[indexer.to_numpy()].mean(axis=0))
        .to_dict()
    )
    final_coords_df = pd.DataFrame(
        {
            "case_id": final_mechanism_df["case_id"].to_numpy(dtype=int),
            "pca1": final_coords[:, 0],
            "pca2": final_coords[:, 1],
        }
    )
    rows: list[dict[str, object]] = []
    for added_index, added_row in added_mechanism_df.iterrows():
        cluster_distances = {
            f"dist_to_cluster_{int(cluster_id)}": float(
                np.sqrt(np.sum(np.square(added_scaled[added_index] - centroid)))
            )
            for cluster_id, centroid in cluster_centroids.items()
        }
        nearest_block = nearest_df[nearest_df["added_case_id"] == added_row["case_id"]].copy()
        first_neighbor = nearest_block.iloc[0]
        neighbor_coord = final_coords_df.loc[
            final_coords_df["case_id"] == int(first_neighbor["neighbor_case_id"])
        ].iloc[0]
        rows.append(
            {
                "case_id": int(added_row["case_id"]),
                "wind_speed": float(added_row["wind_speed"]),
                "rpm": float(added_row["rpm"]),
                "pca1": float(added_coords[added_index, 0]),
                "pca2": float(added_coords[added_index, 1]),
                "nearest_case_id": int(first_neighbor["neighbor_case_id"]),
                "nearest_distance": float(first_neighbor["distance"]),
                "nearest_cluster_id": int(first_neighbor["neighbor_cluster_id"]),
                "nearest_case_wind_speed": float(first_neighbor["neighbor_wind_speed"]),
                "nearest_case_rpm": float(first_neighbor["neighbor_rpm"]),
                "pca_distance_to_nearest": float(
                    np.sqrt(
                        (added_coords[added_index, 0] - float(neighbor_coord["pca1"])) ** 2
                        + (added_coords[added_index, 1] - float(neighbor_coord["pca2"])) ** 2
                    )
                ),
                **cluster_distances,
            }
        )
    diagnostics_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    cluster_cols = [column for column in diagnostics_df.columns if column.startswith("dist_to_cluster_")]
    diagnostics_df["closer_cluster_id"] = diagnostics_df[cluster_cols].idxmin(axis=1).str.replace("dist_to_cluster_", "").astype(int)
    return diagnostics_df


def build_baseline_comparison() -> pd.DataFrame:
    final_manifest_df = pd.read_csv(REPO_ROOT / "data" / "final" / "dataset_manifest.csv")
    final_manifest_df["wind_speed"] = pd.to_numeric(final_manifest_df["wind_speed"], errors="coerce")
    final_manifest_df["rpm"] = pd.to_numeric(final_manifest_df["rpm"], errors="coerce")
    final_manifest_df = final_manifest_df.dropna(subset=["wind_speed", "rpm"]).copy()
    rpm_model = LinearRegression().fit(
        final_manifest_df[["rpm"]].to_numpy(dtype=float),
        final_manifest_df["wind_speed"].to_numpy(dtype=float),
    )

    added_manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
    clean_pred_df = pd.read_csv(ADDED_CLEAN_PRED_PATH)
    full_pred_df = pd.read_csv(ADDED_FULL_PRED_PATH)

    rows: list[dict[str, object]] = []
    for _, row in added_manifest_df.iterrows():
        case_id = int(row["case_id"])
        rpm = float(row["rpm"])
        true_wind_speed = float(row["wind_speed"])
        clean_block = clean_pred_df.loc[clean_pred_df["case_id"] == case_id].iloc[0]
        full_block = full_pred_df.loc[full_pred_df["case_id"] == case_id].iloc[0]
        rpm_linear_pred = float(rpm_model.predict(np.array([[rpm]], dtype=float))[0])
        rpm_knn_pred = compute_rpm_knn_prediction(final_manifest_df, rpm)
        rows.append(
            {
                "case_id": case_id,
                "true_wind_speed": true_wind_speed,
                "rpm": rpm,
                "rpm_linear_pred": rpm_linear_pred,
                "rpm_linear_abs_error": abs(rpm_linear_pred - true_wind_speed),
                "rpm_knn3_pred": rpm_knn_pred,
                "rpm_knn3_abs_error": abs(rpm_knn_pred - true_wind_speed),
                "clean_pred_5s": float(clean_block["pred_5s"]),
                "clean_abs_error_5s": float(clean_block["abs_error_5s"]),
                "clean_pred_fusion": float(clean_block["pred_2s_8s_fusion"]),
                "clean_abs_error_fusion": float(clean_block["abs_error_2s_8s_fusion"]),
                "full_pred_5s": float(full_block["pred_5s"]),
                "full_abs_error_5s": float(full_block["abs_error_5s"]),
                "full_pred_fusion": float(full_block["pred_2s_8s_fusion"]),
                "full_abs_error_fusion": float(full_block["abs_error_2s_8s_fusion"]),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def compute_rpm_knn_prediction(final_manifest_df: pd.DataFrame, rpm: float, neighbor_count: int = 3) -> float:
    block = final_manifest_df.assign(rpm_distance=(final_manifest_df["rpm"] - rpm).abs()).sort_values("rpm_distance").head(neighbor_count)
    weights = 1.0 / np.maximum(block["rpm_distance"].to_numpy(dtype=float), 1.0)
    return float(np.average(block["wind_speed"].to_numpy(dtype=float), weights=weights))


def build_case22_reference_comparison(
    mechanism_df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
) -> pd.DataFrame:
    key_features = outlier_df.loc[outlier_df["case_id"] == 22, "feature_name"].head(6).tolist()
    fixed_features = [
        "wind_speed",
        "rpm",
        "strain_rms_median",
        "acc_rms_median",
        "strain_peak_freq_median",
        "acc_peak_freq_median",
        "strain_low_ratio_median",
        "strain_mid_ratio_median",
        "acc_low_ratio_median",
        "acc_mid_ratio_median",
        "duration_seconds",
        "windows_total",
    ]
    feature_order = []
    for feature_name in [*fixed_features, *key_features]:
        if feature_name in mechanism_df.columns and feature_name not in feature_order:
            feature_order.append(feature_name)

    nearest_case_ids = nearest_df.loc[nearest_df["added_case_id"] == 22, "neighbor_case_id"].head(3).tolist()
    reference_ids = [22, *nearest_case_ids]
    reference_df = mechanism_df.loc[mechanism_df["case_id"].isin(reference_ids), ["case_id", *feature_order]].copy()
    reference_df["reference_role"] = reference_df["case_id"].map(
        {
            22: "added_case22",
            **{case_id: f"nearest_final_rank_{rank}" for rank, case_id in enumerate(nearest_case_ids, start=1)},
        }
    )
    columns = ["reference_role", "case_id", *feature_order]
    return reference_df[columns].sort_values(["reference_role", "case_id"]).reset_index(drop=True)


def create_projection_scatter(
    final_mechanism_df: pd.DataFrame,
    added_mechanism_df: pd.DataFrame,
    final_coords: np.ndarray,
    added_coords: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    cmap = plt.get_cmap("tab10")

    for cluster_id in sorted(final_mechanism_df["cluster_id"].dropna().unique()):
        cluster_mask = final_mechanism_df["cluster_id"] == cluster_id
        ax.scatter(
            final_coords[cluster_mask.to_numpy(), 0],
            final_coords[cluster_mask.to_numpy(), 1],
            s=80,
            alpha=0.75,
            color=cmap(int(cluster_id)),
            label=f"final cluster {int(cluster_id)}",
        )

    for row_index, row in final_mechanism_df.iterrows():
        ax.annotate(
            f"F{int(row['case_id'])}",
            (final_coords[row_index, 0], final_coords[row_index, 1]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax.scatter(
        added_coords[:, 0],
        added_coords[:, 1],
        marker="*",
        s=260,
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
        label="added 21-24",
        zorder=5,
    )
    for row_index, row in added_mechanism_df.iterrows():
        ax.annotate(
            f"A{int(row['case_id'])}",
            (added_coords[row_index, 0], added_coords[row_index, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Added cases projected into final mechanism PCA space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_case22_spectrum_plot(
    records_by_case: dict[int, DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    nearest_df: pd.DataFrame,
    common_signal_columns: list[str],
    output_path: Path,
) -> None:
    case22_neighbors = nearest_df.loc[nearest_df["added_case_id"] == 22, "neighbor_case_id"].head(2).tolist()
    compare_case_ids = [22, *case22_neighbors]
    spectrum_map = {
        case_id: compute_case_spectrum_summary(
            cleaned_signal_frames[case_id],
            strain_columns=[column for column in common_signal_columns if "应变" in column],
            acc_columns=[column for column in common_signal_columns if "Acc" in column],
        )
        for case_id in compare_case_ids
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for case_id in compare_case_ids:
        label = f"case{case_id}"
        if case_id == 22:
            label += " (added)"
        axes[0].plot(
            spectrum_map[case_id]["strain_freqs"],
            spectrum_map[case_id]["strain_power"],
            label=label,
            linewidth=2.2 if case_id == 22 else 1.6,
        )
        axes[1].plot(
            spectrum_map[case_id]["acc_freqs"],
            spectrum_map[case_id]["acc_power"],
            label=label,
            linewidth=2.2 if case_id == 22 else 1.6,
        )

    axes[0].set_xlim(0.0, 10.0)
    axes[1].set_xlim(0.0, 15.0)
    axes[0].set_title("Average normalized strain spectrum")
    axes[1].set_title("Average normalized acceleration spectrum")
    axes[0].set_xlabel("Hz")
    axes[1].set_xlabel("Hz")
    axes[0].set_ylabel("relative power")
    axes[1].set_ylabel("relative power")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_case_spectrum_summary(
    cleaned_df: pd.DataFrame,
    strain_columns: list[str],
    acc_columns: list[str],
    sampling_rate: float = 50.0,
) -> dict[str, np.ndarray]:
    def average_relative_power(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        spectra: list[np.ndarray] = []
        freqs: np.ndarray | None = None
        for column in columns:
            signal = cleaned_df[column].to_numpy(dtype=float, copy=False)
            centered = signal - float(np.mean(signal))
            power = np.square(np.abs(np.fft.rfft(centered)))
            if power.size > 1:
                power[0] = 0.0
            total = float(power.sum())
            spectra.append(power / total if total > 0 else power)
            if freqs is None:
                freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
        if freqs is None:
            raise ValueError("无法从空通道集合构建频谱。")
        return freqs, np.mean(np.vstack(spectra), axis=0)

    strain_freqs, strain_power = average_relative_power(strain_columns)
    acc_freqs, acc_power = average_relative_power(acc_columns)
    return {
        "strain_freqs": strain_freqs,
        "strain_power": strain_power,
        "acc_freqs": acc_freqs,
        "acc_power": acc_power,
    }


def write_summary_markdown(
    output_path: Path,
    common_signal_columns: list[str],
    final_records: list[DatasetRecord],
    diagnostics_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
) -> None:
    case22_outliers = outlier_df.loc[outlier_df["case_id"] == 22].head(5)
    lines = [
        "# added 反常表现域诊断",
        "",
        f"- `final` 单独共有通道数：`{len(get_common_signal_columns(final_records))}`",
        f"- `final + added` 共有通道数：`{len(common_signal_columns)}`",
        "- 当前诊断基于 `034/035` 已有外部预测结果，不重复重训主模型。",
        "",
        "## added 工况最近邻",
        "",
    ]

    for _, row in diagnostics_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: nearest=`工况{int(row['nearest_case_id'])}` (dist=`{row['nearest_distance']:.3f}`), closer_cluster=`{int(row['closer_cluster_id'])}`"
        )

    lines.extend(["", "## 模型 / 基线对照", ""])
    for _, row in baseline_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.2f}`, rpm_linear=`{row['rpm_linear_pred']:.3f}`, clean_5s=`{row['clean_pred_5s']:.3f}`, full_5s=`{row['full_pred_5s']:.3f}`"
        )

    lines.extend(["", "## 工况22 主要异常特征", ""])
    for _, row in case22_outliers.iterrows():
        lines.append(
            f"- `{row['feature_name']}`: value=`{row['feature_value']:.6g}`, z=`{row['z_score_vs_final']:+.3f}`"
        )

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- `added` 没有引入通道缺失，异常更像机制域偏移，不像输入列口径被改坏。",
            "- `工况22` 在机制空间里与所有 `final` 工况都很远，应优先当作单独异常机制点复核。",
            "- `rpm-only` 基线对 `工况21/23/24` 明显更接近标签，这说明当前高估主要来自信号域外推，而不是 RPM 标签本身全体失真。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
