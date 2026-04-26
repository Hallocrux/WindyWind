from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY030_ROOT = REPO_ROOT / "src" / "try" / "030_case_mechanism_clustering"
TRY0431_ROOT = REPO_ROOT / "src" / "try" / "043_1_fft_rpm_algorithm_search"
for path in (REPO_ROOT, REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import (  # noqa: E402
    DatasetRecord,
    build_dataset_inventory,
    get_common_signal_columns,
    prepare_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig, build_case_feature_frame  # noqa: E402

TRY_NAME = "062_added2_domain_diagnosis"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_STANDARD_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
TOP_OUTLIER_COUNT = 6
NEIGHBOR_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="added2 新补充工况域诊断。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try030 = _load_module(
        TRY030_ROOT / "run_case_mechanism_clustering.py",
        module_name="try030_module",
    )
    try0431 = _load_module(
        TRY0431_ROOT / "run_fft_rpm_algorithm_search.py",
        module_name="try0431_module",
    )

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_manifest_records(ADDED_MANIFEST_PATH, ADDED_STANDARD_DIR)
    added2_records = load_manifest_records(ADDED2_MANIFEST_PATH, ADDED2_STANDARD_DIR)
    reference_records = [*final_records, *added_records]
    all_records = [*reference_records, *added2_records]

    common_signal_columns = get_common_signal_columns(all_records)
    strain_channels = [column for column in common_signal_columns if "应变" in column]
    acc_channels = [column for column in common_signal_columns if "Acc" in column]

    cleaned_frames: dict[int, pd.DataFrame] = {}
    mechanism_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    window_totals: dict[int, int] = {}
    record_domain = build_domain_map(final_records, added_records, added2_records)

    for record in all_records:
        cleaned, stats = prepare_clean_signal_frame(record, common_signal_columns)
        cleaned_frames[record.case_id] = cleaned
        feature_df = build_case_feature_frame(record, cleaned, WindowConfig())
        window_totals[record.case_id] = int(len(feature_df))

        row = {
            "case_id": record.case_id,
            "domain": record_domain[record.case_id],
            "display_name": record.display_name,
            "file_name": record.file_name,
            "wind_speed": record.wind_speed,
            "rpm": record.rpm,
        }
        row.update(
            try030.summarize_case_windows(
                feature_df=feature_df,
                strain_channels=strain_channels,
                acc_channels=acc_channels,
            )
        )
        mechanism_rows.append(row)

        quality_rows.append(
            build_quality_row(
                record=record,
                common_signal_columns=common_signal_columns,
                stats=stats,
            )
        )

    inventory_df = build_dataset_inventory(all_records)
    mechanism_df = pd.DataFrame(mechanism_rows)
    quality_df = pd.DataFrame(quality_rows)
    mechanism_df = mechanism_df.merge(quality_df, on=["case_id", "domain"], how="left")
    mechanism_df = mechanism_df.merge(
        inventory_df[["case_id", "row_count", "column_count", "duration_seconds", "sampling_hz_est"]],
        on="case_id",
        how="left",
    )
    mechanism_df["windows_total"] = mechanism_df["case_id"].map(window_totals)
    mechanism_df = mechanism_df.sort_values(["domain", "case_id"]).reset_index(drop=True)

    feature_columns = try030.get_mechanism_feature_columns()
    reference_df = mechanism_df[mechanism_df["domain"].isin(["final", "added"])].copy().reset_index(drop=True)
    added2_df = mechanism_df[mechanism_df["domain"] == "added2"].copy().reset_index(drop=True)

    scaler_all = StandardScaler()
    reference_scaled = scaler_all.fit_transform(reference_df[feature_columns].to_numpy(dtype=float))
    added2_scaled = scaler_all.transform(added2_df[feature_columns].to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=42)
    reference_coords = pca.fit_transform(reference_scaled)
    added2_coords = pca.transform(added2_scaled)

    scaler_final = StandardScaler()
    final_feature_df = reference_df[reference_df["domain"] == "final"].copy().reset_index(drop=True)
    final_scaled = scaler_final.fit_transform(final_feature_df[feature_columns].to_numpy(dtype=float))
    added2_scaled_vs_final = scaler_final.transform(added2_df[feature_columns].to_numpy(dtype=float))

    nearest_df = build_nearest_reference_table(reference_df, added2_df, reference_scaled, added2_scaled)
    domain_summary_df = build_domain_summary(
        reference_df=reference_df,
        added2_df=added2_df,
        reference_scaled=reference_scaled,
        added2_scaled=added2_scaled,
        nearest_df=nearest_df,
        reference_coords=reference_coords,
        added2_coords=added2_coords,
    )
    outlier_df = build_outlier_table(
        added2_df=added2_df,
        feature_columns=feature_columns,
        added2_scaled_vs_final=added2_scaled_vs_final,
    )
    consistency_df = build_label_rpm_consistency_table(
        try0431=try0431,
        final_records=final_records,
        added2_records=added2_records,
        cleaned_frames=cleaned_frames,
    )

    case_inventory_df = mechanism_df[
        [
            "case_id",
            "domain",
            "display_name",
            "wind_speed",
            "rpm",
            "row_count",
            "column_count",
            "duration_seconds",
            "sampling_hz_est",
            "missing_ratio_in_common_cols",
            "edge_removed_ratio",
            "rows_after_edge_drop",
            "rows_after_long_gap_drop",
            "continuous_segment_count",
            "windows_total",
        ]
    ].sort_values(["domain", "case_id"]).reset_index(drop=True)

    case_inventory_df.to_csv(output_dir / "case_inventory.csv", index=False, encoding="utf-8-sig")
    nearest_df.to_csv(output_dir / "nearest_reference_cases.csv", index=False, encoding="utf-8-sig")
    domain_summary_df.to_csv(output_dir / "added2_domain_summary.csv", index=False, encoding="utf-8-sig")
    outlier_df.to_csv(output_dir / "added2_feature_outliers_vs_final.csv", index=False, encoding="utf-8-sig")
    consistency_df.to_csv(output_dir / "label_rpm_consistency.csv", index=False, encoding="utf-8-sig")

    create_projection_plot(
        reference_df=reference_df,
        added2_df=added2_df,
        reference_coords=reference_coords,
        added2_coords=added2_coords,
        output_path=output_dir / "mechanism_projection.png",
    )
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        case_inventory_df=case_inventory_df,
        domain_summary_df=domain_summary_df,
        outlier_df=outlier_df,
        consistency_df=consistency_df,
        pca=pca,
    )

    print("added2 域诊断完成。")
    print(f"输出目录: {output_dir}")


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_manifest_records(manifest_path: Path, data_dir: Path) -> list[DatasetRecord]:
    manifest_df = pd.read_csv(manifest_path)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        wind_speed = pd.to_numeric(pd.Series([row["wind_speed"]]), errors="coerce").iloc[0]
        rpm = pd.to_numeric(pd.Series([row["rpm"]]), errors="coerce").iloc[0]
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=data_dir / f"工况{case_id}.csv",
                wind_speed=float(wind_speed) if pd.notna(wind_speed) else None,
                rpm=float(rpm) if pd.notna(rpm) else None,
                is_labeled=pd.notna(wind_speed) and pd.notna(rpm),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return [record for record in records if record.is_labeled]


def build_domain_map(
    final_records: list[DatasetRecord],
    added_records: list[DatasetRecord],
    added2_records: list[DatasetRecord],
) -> dict[int, str]:
    domain_map: dict[int, str] = {}
    for record in final_records:
        domain_map[record.case_id] = "final"
    for record in added_records:
        domain_map[record.case_id] = "added"
    for record in added2_records:
        domain_map[record.case_id] = "added2"
    return domain_map


def build_quality_row(
    record: DatasetRecord,
    common_signal_columns: list[str],
    stats,
) -> dict[str, object]:
    raw_df = pd.read_csv(record.file_path)
    numeric_df = raw_df[common_signal_columns].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(numeric_df.isna().sum().sum())
    total_cells = max(int(len(raw_df) * len(common_signal_columns)), 1)
    return {
        "case_id": record.case_id,
        "domain": "final" if record.case_id <= 20 else ("added" if record.case_id <= 24 else "added2"),
        "missing_ratio_in_common_cols": float(missing_cells / total_cells),
        "edge_removed_ratio": float(stats.edge_removed_rows / max(len(raw_df), 1)),
        "rows_after_edge_drop": int(stats.rows_after_edge_drop),
        "rows_after_long_gap_drop": int(stats.rows_after_long_gap_drop),
        "continuous_segment_count": int(stats.continuous_segment_count),
    }


def build_nearest_reference_table(
    reference_df: pd.DataFrame,
    added2_df: pd.DataFrame,
    reference_scaled: np.ndarray,
    added2_scaled: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for added2_index, added2_row in added2_df.iterrows():
        distances = np.sqrt(np.sum(np.square(reference_scaled - added2_scaled[added2_index]), axis=1))
        order = np.argsort(distances)[:NEIGHBOR_COUNT]
        for rank, ref_index in enumerate(order, start=1):
            ref_row = reference_df.iloc[ref_index]
            rows.append(
                {
                    "added2_case_id": int(added2_row["case_id"]),
                    "neighbor_rank": rank,
                    "neighbor_case_id": int(ref_row["case_id"]),
                    "neighbor_domain": str(ref_row["domain"]),
                    "distance": float(distances[ref_index]),
                    "neighbor_wind_speed": float(ref_row["wind_speed"]),
                    "neighbor_rpm": float(ref_row["rpm"]),
                    "neighbor_display_name": str(ref_row["display_name"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["added2_case_id", "neighbor_rank"]).reset_index(drop=True)


def build_domain_summary(
    reference_df: pd.DataFrame,
    added2_df: pd.DataFrame,
    reference_scaled: np.ndarray,
    added2_scaled: np.ndarray,
    nearest_df: pd.DataFrame,
    reference_coords: np.ndarray,
    added2_coords: np.ndarray,
) -> pd.DataFrame:
    reference_coords_df = pd.DataFrame(
        {
            "case_id": reference_df["case_id"].to_numpy(dtype=int),
            "pca1": reference_coords[:, 0],
            "pca2": reference_coords[:, 1],
            "domain": reference_df["domain"].tolist(),
        }
    )
    centroid_map = {
        domain_name: reference_scaled[reference_df["domain"] == domain_name].mean(axis=0)
        for domain_name in sorted(reference_df["domain"].unique())
    }
    rows: list[dict[str, object]] = []
    for added2_index, added2_row in added2_df.iterrows():
        nearest_block = nearest_df[nearest_df["added2_case_id"] == int(added2_row["case_id"])].copy()
        first_neighbor = nearest_block.iloc[0]
        nearest_counts = nearest_block["neighbor_domain"].value_counts().to_dict()
        nearest_final = nearest_block[nearest_block["neighbor_domain"] == "final"].head(1)
        nearest_added = nearest_block[nearest_block["neighbor_domain"] == "added"].head(1)
        centroid_distances = {
            f"dist_to_{domain_name}_centroid": float(
                np.sqrt(np.sum(np.square(added2_scaled[added2_index] - centroid)))
            )
            for domain_name, centroid in centroid_map.items()
        }
        ref_coord = reference_coords_df.loc[
            reference_coords_df["case_id"] == int(first_neighbor["neighbor_case_id"])
        ].iloc[0]
        rows.append(
            {
                "case_id": int(added2_row["case_id"]),
                "display_name": str(added2_row["display_name"]),
                "wind_speed": float(added2_row["wind_speed"]),
                "rpm": float(added2_row["rpm"]),
                "pca1": float(added2_coords[added2_index, 0]),
                "pca2": float(added2_coords[added2_index, 1]),
                "nearest_case_id": int(first_neighbor["neighbor_case_id"]),
                "nearest_domain": str(first_neighbor["neighbor_domain"]),
                "nearest_distance": float(first_neighbor["distance"]),
                "nearest_final_case_id": int(nearest_final.iloc[0]["neighbor_case_id"]) if not nearest_final.empty else np.nan,
                "nearest_final_distance": float(nearest_final.iloc[0]["distance"]) if not nearest_final.empty else np.nan,
                "nearest_added_case_id": int(nearest_added.iloc[0]["neighbor_case_id"]) if not nearest_added.empty else np.nan,
                "nearest_added_distance": float(nearest_added.iloc[0]["distance"]) if not nearest_added.empty else np.nan,
                "top4_final_count": int(nearest_counts.get("final", 0)),
                "top4_added_count": int(nearest_counts.get("added", 0)),
                "pca_distance_to_nearest": float(
                    np.sqrt(
                        (added2_coords[added2_index, 0] - float(ref_coord["pca1"])) ** 2
                        + (added2_coords[added2_index, 1] - float(ref_coord["pca2"])) ** 2
                    )
                ),
                **centroid_distances,
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    centroid_cols = [column for column in summary_df.columns if column.startswith("dist_to_")]
    summary_df["closer_centroid_domain"] = (
        summary_df[centroid_cols].idxmin(axis=1).str.replace("dist_to_", "", regex=False).str.replace("_centroid", "", regex=False)
    )
    return summary_df


def build_outlier_table(
    added2_df: pd.DataFrame,
    feature_columns: list[str],
    added2_scaled_vs_final: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for added2_index, added2_row in added2_df.iterrows():
        z_values = added2_scaled_vs_final[added2_index]
        order = np.argsort(np.abs(z_values))[::-1][:TOP_OUTLIER_COUNT]
        for rank, feature_index in enumerate(order, start=1):
            feature_name = feature_columns[feature_index]
            rows.append(
                {
                    "case_id": int(added2_row["case_id"]),
                    "outlier_rank": rank,
                    "feature_name": feature_name,
                    "feature_value": float(added2_row[feature_name]),
                    "z_score_vs_final": float(z_values[feature_index]),
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "outlier_rank"]).reset_index(drop=True)


def build_label_rpm_consistency_table(
    try0431,
    final_records: list[DatasetRecord],
    added2_records: list[DatasetRecord],
    cleaned_frames: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    final_manifest_df = pd.DataFrame(
        [{"case_id": record.case_id, "wind_speed": record.wind_speed, "rpm": record.rpm} for record in final_records]
    )
    whole_spec = try0431.VariantSpec("fft_peak_1x_whole", estimator="peak_1x", spectrum_mode="whole")
    window8_spec = try0431.VariantSpec(
        "window_peak_1x_conf_8s",
        estimator="peak_1x",
        spectrum_mode="whole",
        window_estimator_seconds=8.0,
    )

    rows: list[dict[str, object]] = []
    for record in added2_records:
        frame = cleaned_frames[record.case_id]
        whole_estimate = try0431.estimate_record_rpm(record=record, frame=frame, spec=whole_spec)
        window8_estimate = try0431.estimate_record_rpm(record=record, frame=frame, spec=window8_spec)
        hybrid_rpm = choose_hybrid_rpm(
            pred_whole=whole_estimate.pred_rpm,
            pred_window8=window8_estimate.pred_rpm,
        )
        pred_wind_from_label = predict_rpm_knn4(final_manifest_df, float(record.rpm))
        pred_wind_from_whole = predict_rpm_knn4(final_manifest_df, float(whole_estimate.pred_rpm))
        pred_wind_from_hybrid = predict_rpm_knn4(final_manifest_df, float(hybrid_rpm))
        rows.append(
            {
                "case_id": record.case_id,
                "display_name": record.display_name,
                "true_wind_speed": float(record.wind_speed),
                "label_rpm": float(record.rpm),
                "wind_pred_from_label_rpm_knn4": pred_wind_from_label,
                "wind_abs_error_from_label_rpm_knn4": abs(pred_wind_from_label - float(record.wind_speed)),
                "fft_peak_1x_whole_rpm": float(whole_estimate.pred_rpm),
                "fft_peak_1x_whole_abs_error_rpm": abs(float(whole_estimate.pred_rpm) - float(record.rpm)),
                "window_peak_1x_conf_8s_rpm": float(window8_estimate.pred_rpm),
                "window_peak_1x_conf_8s_abs_error_rpm": abs(float(window8_estimate.pred_rpm) - float(record.rpm)),
                "hybrid_gate150_rpm": float(hybrid_rpm),
                "hybrid_gate150_abs_error_rpm": abs(float(hybrid_rpm) - float(record.rpm)),
                "wind_pred_from_fft_peak_knn4": pred_wind_from_whole,
                "wind_abs_error_from_fft_peak_knn4": abs(pred_wind_from_whole - float(record.wind_speed)),
                "wind_pred_from_hybrid_knn4": pred_wind_from_hybrid,
                "wind_abs_error_from_hybrid_knn4": abs(pred_wind_from_hybrid - float(record.wind_speed)),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def choose_hybrid_rpm(*, pred_whole: float, pred_window8: float) -> float:
    if np.isclose(pred_whole, pred_window8, atol=1e-9):
        return float(pred_whole)
    use_window = max(pred_whole, pred_window8) < 150.0 or pred_window8 > pred_whole
    return float(pred_window8 if use_window else pred_whole)


def predict_rpm_knn4(train_df: pd.DataFrame, rpm_value: float, neighbor_count: int = 4) -> float:
    block = train_df.assign(rpm_distance=(train_df["rpm"] - rpm_value).abs()).sort_values("rpm_distance").head(neighbor_count)
    weights = 1.0 / np.maximum(block["rpm_distance"].to_numpy(dtype=float), 1.0)
    return float(np.average(block["wind_speed"].to_numpy(dtype=float), weights=weights))


def create_projection_plot(
    reference_df: pd.DataFrame,
    added2_df: pd.DataFrame,
    reference_coords: np.ndarray,
    added2_coords: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    palette = {
        "final": "#1f77b4",
        "added": "#ff7f0e",
    }
    markers = {
        "final": "o",
        "added": "^",
        "added2": "s",
    }

    for domain_name in ("final", "added"):
        mask = reference_df["domain"] == domain_name
        ax.scatter(
            reference_coords[mask.to_numpy(), 0],
            reference_coords[mask.to_numpy(), 1],
            s=70 if domain_name == "final" else 100,
            alpha=0.75,
            marker=markers[domain_name],
            color=palette[domain_name],
            label=domain_name,
        )

    for row_index, row in reference_df.iterrows():
        prefix = "F" if row["domain"] == "final" else "A"
        ax.annotate(
            f"{prefix}{int(row['case_id'])}",
            (reference_coords[row_index, 0], reference_coords[row_index, 1]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    ax.scatter(
        added2_coords[:, 0],
        added2_coords[:, 1],
        s=180,
        marker=markers["added2"],
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
        label="added2",
        zorder=5,
    )
    for row_index, row in added2_df.iterrows():
        ax.annotate(
            f"N{int(row['case_id'])}",
            (added2_coords[row_index, 0], added2_coords[row_index, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("added2 projected into final + added mechanism space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    case_inventory_df: pd.DataFrame,
    domain_summary_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    pca: PCA,
) -> None:
    added2_inventory = case_inventory_df[case_inventory_df["domain"] == "added2"].copy()
    lines = [
        "# added2 域诊断摘要",
        "",
        f"- 诊断日期：`{pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d')}`",
        f"- added2 工况数：`{len(added2_inventory)}`",
        f"- 参考池工况数：`{len(case_inventory_df[case_inventory_df['domain'].isin(['final', 'added'])])}`",
        f"- 统一空间 PCA explained variance：`PC1={pca.explained_variance_ratio_[0]:.2%}`, `PC2={pca.explained_variance_ratio_[1]:.2%}`",
        "",
        "## added2 邻域摘要",
        "",
    ]
    for _, row in domain_summary_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: nearest=`工况{int(row['nearest_case_id'])}`({row['nearest_domain']}), "
            f"top4 final/added=`{int(row['top4_final_count'])}/{int(row['top4_added_count'])}`, "
            f"closer_centroid=`{row['closer_centroid_domain']}`"
        )

    lines.extend(["", "## 标签与 RPM 自洽性", ""])
    for _, row in consistency_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: label_rpm=`{row['label_rpm']:.1f}`, "
            f"fft_whole_err=`{row['fft_peak_1x_whole_abs_error_rpm']:.1f}`, "
            f"hybrid_err=`{row['hybrid_gate150_abs_error_rpm']:.1f}`, "
            f"label_rpm->wind_knn4_err=`{row['wind_abs_error_from_label_rpm_knn4']:.3f}`"
        )

    lines.extend(["", "## added2 相对 final 的主要异常特征", ""])
    for case_id in consistency_df["case_id"].tolist():
        case_block = outlier_df[outlier_df["case_id"] == case_id].head(3)
        features = ", ".join(
            f"{row['feature_name']}({row['z_score_vs_final']:+.2f})"
            for _, row in case_block.iterrows()
        )
        lines.append(f"- `工况{int(case_id)}`: {features}")

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- added2 更适合作为外部域诊断、route/gate 校准和 reference pool 扩充候选，而不是直接并入默认训练主线。",
            "- 是否纳入监督训练，应先看它与旧 added 的相似度是否稳定、以及 FFT/RPM 链与截图标签是否一致。",
            "- 若要最大化利用 added2，优先顺序应是：先扩验证与诊断，再扩 reference/gate，最后才讨论并池重训。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
