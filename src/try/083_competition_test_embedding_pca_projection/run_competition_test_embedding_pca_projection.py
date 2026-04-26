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
    build_embedding_case_table,
    build_record_table,
    get_embedding_columns,
    load_fixed_window_embeddings,
    load_source_catalog,
    load_try053_module,
)
from src.current.data_loading import DatasetRecord, get_common_signal_columns, prepare_clean_signal_frame  # noqa: E402

TRY_NAME = "083_competition_test_embedding_pca_projection"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
DOMAIN_COLORS = {
    "final": "#4c78a8",
    "added": "#f58518",
    "added2": "#e45756",
}
TEST_COLORS = {
    "test_01_rpm204": "#2e8b57",
    "test_02_rpm192": "#d95f02",
    "test_02_rpm228": "#7570b3",
    "test_04_rpm250": "#1b9e77",
}
TEST_SPECS = [
    {
        "case_id": 1001,
        "display_name": "test_01_rpm204",
        "file_name": "竞赛预测风速工况.csv",
        "file_path": REPO_ROOT / "data" / "test" / "竞赛预测风速工况.csv",
        "rpm": 204.0,
        "label": "T1-204",
    },
    {
        "case_id": 1002,
        "display_name": "test_02_rpm192",
        "file_name": "竞赛预测风速工况2 .csv",
        "file_path": REPO_ROOT / "data" / "test" / "竞赛预测风速工况2 .csv",
        "rpm": 192.0,
        "label": "T2-192",
    },
    {
        "case_id": 1003,
        "display_name": "test_02_rpm228",
        "file_name": "竞赛预测风速工况2 .csv",
        "file_path": REPO_ROOT / "data" / "test" / "竞赛预测风速工况2 .csv",
        "rpm": 228.0,
        "label": "T2-228",
    },
    {
        "case_id": 1004,
        "display_name": "test_04_rpm250",
        "file_name": "竞赛预测风速工况4.csv",
        "file_path": REPO_ROOT / "data" / "test" / "竞赛预测风速工况4.csv",
        "rpm": 250.0,
        "label": "T4-250",
    },
]
TEXT_OFFSETS = {
    "T1-204": (0.10, 0.08),
    "T2-192": (0.10, -0.12),
    "T2-228": (0.10, 0.12),
    "T4-250": (0.10, 0.08),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把四组竞赛测试输入投影到 TinyTCN 2s+8s embedding PCA 平面。")
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
    base_record_df = build_record_table(catalog)
    test_records = build_test_records()

    export_records = [*catalog.all_records, *test_records]
    common_signal_columns = get_common_signal_columns(export_records)
    cleaned_signal_frames = build_cleaned_signal_frames(export_records, common_signal_columns)

    fixed_window_embeddings = load_fixed_window_embeddings(
        try053=try053,
        export_records=export_records,
        cleaned_signal_frames=cleaned_signal_frames,
    )

    test_record_df = build_test_record_table(test_records)
    record_df = pd.concat([base_record_df, test_record_df], ignore_index=True)
    embedding_case_df = build_embedding_case_table(record_df, fixed_window_embeddings)
    embedding_columns = get_embedding_columns(embedding_case_df)
    pca_df, pca = build_pca_projection(embedding_case_df, embedding_columns)

    competition_summary_df = (
        pca_df.loc[pca_df["raw_source_domain"] == "competition_test"]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    competition_summary_df["shared_signal_group"] = competition_summary_df["file_name"]

    embedding_case_df.to_csv(output_dir / "embedding_case_table.csv", index=False, encoding="utf-8-sig")
    pca_df.to_csv(output_dir / "embedding_pca_coords.csv", index=False, encoding="utf-8-sig")
    competition_summary_df.to_csv(output_dir / "competition_test_projection_summary.csv", index=False, encoding="utf-8-sig")

    create_pca_with_competition_tests_plot(pca_df, plot_dir / "pca_with_competition_tests.png")
    create_competition_focus_plot(pca_df, plot_dir / "pca_competition_focus.png")
    create_projection_panel_plot(pca_df, plot_dir / "pca_competition_projection_panel.png")
    write_summary_markdown(output_dir / "summary.md", pca_df=pca_df, pca=pca)

    print("083 competition test embedding PCA projection 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")


def build_test_records() -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for spec in TEST_SPECS:
        if not Path(spec["file_path"]).exists():
            raise FileNotFoundError(f"未找到测试文件: {spec['file_path']}")
        records.append(
            DatasetRecord(
                case_id=int(spec["case_id"]),
                display_name=str(spec["display_name"]),
                file_name=str(spec["file_name"]),
                file_path=Path(spec["file_path"]),
                wind_speed=None,
                rpm=float(spec["rpm"]),
                is_labeled=False,
                original_file_name=str(spec["file_name"]),
                label_source="user_provided_rpm_2026-04-09",
                notes=str(spec["label"]),
            )
        )
    return records


def build_cleaned_signal_frames(
    records: list[DatasetRecord],
    common_signal_columns: list[str],
) -> dict[int, pd.DataFrame]:
    cleaned_frames: dict[int, pd.DataFrame] = {}
    for record in records:
        cleaned_frame, _ = prepare_clean_signal_frame(record, common_signal_columns)
        cleaned_frames[record.case_id] = cleaned_frame
    return cleaned_frames


def build_test_record_table(test_records: list[DatasetRecord]) -> pd.DataFrame:
    rows = []
    for record, spec in zip(test_records, TEST_SPECS):
        rows.append(
            {
                "case_id": int(record.case_id),
                "file_name": str(record.file_name),
                "display_name": str(record.display_name),
                "raw_source_domain": "competition_test",
                "wind_speed": np.nan,
                "rpm": float(record.rpm),
                "is_labeled": False,
                "notes": str(record.notes),
                "plot_label": str(spec["label"]),
                "source_path": str(record.file_path),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_pca_projection(
    embedding_case_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, PCA]:
    matrix = embedding_case_df[embedding_columns].to_numpy(dtype=float, copy=False)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    keep_columns = [
        column
        for column in [
            "case_id",
            "file_name",
            "display_name",
            "raw_source_domain",
            "wind_speed",
            "rpm",
            "is_labeled",
            "notes",
            "plot_label",
            "source_path",
        ]
        if column in embedding_case_df.columns
    ]
    pca_df = embedding_case_df[keep_columns].copy()
    pca_df["pca1"] = coords[:, 0]
    pca_df["pca2"] = coords[:, 1]
    return pca_df, pca


def create_pca_with_competition_tests_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    for domain_name in ("final", "added", "added2"):
        block = pca_df.loc[pca_df["raw_source_domain"] == domain_name]
        if block.empty:
            continue
        ax.scatter(
            block["pca1"],
            block["pca2"],
            s=90 if domain_name != "added2" else 120,
            alpha=0.82,
            label=domain_name,
            color=DOMAIN_COLORS[domain_name],
            edgecolors="black" if domain_name == "added2" else "none",
            linewidths=0.8 if domain_name == "added2" else 0.0,
        )

    competition_df = pca_df.loc[pca_df["raw_source_domain"] == "competition_test"].copy()
    for _, row in competition_df.iterrows():
        color = TEST_COLORS.get(str(row["display_name"]), "#222222")
        ax.scatter(
            [row["pca1"]],
            [row["pca2"]],
            s=210,
            alpha=0.98,
            color=color,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
        )
        dx, dy = TEXT_OFFSETS.get(str(row["plot_label"]), (0.08, 0.08))
        ax.text(
            float(row["pca1"]) + dx,
            float(row["pca2"]) + dy,
            str(row["plot_label"]),
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )

    ax.set_title("TinyTCN 2s+8s embedding PCA with competition tests")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_competition_focus_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    background = pca_df.loc[pca_df["raw_source_domain"] != "competition_test"]
    competition_df = pca_df.loc[pca_df["raw_source_domain"] == "competition_test"].copy()

    ax.scatter(
        background["pca1"],
        background["pca2"],
        s=60,
        alpha=0.16,
        color="#a7aeb6",
        label="final + added + added2 background",
    )
    for _, row in competition_df.iterrows():
        color = TEST_COLORS.get(str(row["display_name"]), "#222222")
        ax.scatter(
            [row["pca1"]],
            [row["pca2"]],
            s=250,
            alpha=0.98,
            color=color,
            marker="X",
            edgecolors="black",
            linewidths=1.3,
        )
        dx, dy = TEXT_OFFSETS.get(str(row["plot_label"]), (0.08, 0.08))
        ax.text(
            float(row["pca1"]) + dx,
            float(row["pca2"]) + dy,
            str(row["plot_label"]),
            fontsize=11,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
    ax.set_title("Competition test projection in TinyTCN 2s+8s embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_projection_panel_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    all_ax = axes[0]
    for domain_name in ("final", "added", "added2"):
        block = pca_df.loc[pca_df["raw_source_domain"] == domain_name]
        if block.empty:
            continue
        all_ax.scatter(
            block["pca1"],
            block["pca2"],
            s=85 if domain_name != "added2" else 115,
            alpha=0.82,
            label=domain_name,
            color=DOMAIN_COLORS[domain_name],
            edgecolors="black" if domain_name == "added2" else "none",
            linewidths=0.8 if domain_name == "added2" else 0.0,
        )
    competition_df = pca_df.loc[pca_df["raw_source_domain"] == "competition_test"].copy()
    for _, row in competition_df.iterrows():
        color = TEST_COLORS.get(str(row["display_name"]), "#222222")
        all_ax.scatter(
            [row["pca1"]],
            [row["pca2"]],
            s=210,
            alpha=0.98,
            color=color,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
        )
        dx, dy = TEXT_OFFSETS.get(str(row["plot_label"]), (0.08, 0.08))
        all_ax.text(
            float(row["pca1"]) + dx,
            float(row["pca2"]) + dy,
            str(row["plot_label"]),
            fontsize=9.5,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
    all_ax.set_title("All domains + competition tests")
    all_ax.set_xlabel("PC1")
    all_ax.set_ylabel("PC2")
    all_ax.legend()

    focus_ax = axes[1]
    background = pca_df.loc[pca_df["raw_source_domain"] != "competition_test"]
    focus_ax.scatter(
        background["pca1"],
        background["pca2"],
        s=60,
        alpha=0.16,
        color="#a7aeb6",
        label="background",
    )
    for _, row in competition_df.iterrows():
        color = TEST_COLORS.get(str(row["display_name"]), "#222222")
        focus_ax.scatter(
            [row["pca1"]],
            [row["pca2"]],
            s=250,
            alpha=0.98,
            color=color,
            marker="X",
            edgecolors="black",
            linewidths=1.3,
        )
        dx, dy = TEXT_OFFSETS.get(str(row["plot_label"]), (0.08, 0.08))
        focus_ax.text(
            float(row["pca1"]) + dx,
            float(row["pca2"]) + dy,
            str(row["plot_label"]),
            fontsize=10.5,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
    focus_ax.set_title("Competition test focus")
    focus_ax.set_xlabel("PC1")
    focus_ax.set_ylabel("PC2")
    focus_ax.legend()

    fig.suptitle("TinyTCN 2s+8s embedding PCA projection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(output_path: Path, *, pca_df: pd.DataFrame, pca: PCA) -> None:
    competition_df = pca_df.loc[pca_df["raw_source_domain"] == "competition_test"].sort_values("case_id")
    lines = [
        "# competition test embedding PCA projection",
        "",
        "## 摘要",
        "",
        f"- 导出样本数：`{len(pca_df)}`",
        f"- `final` 工况数：`{int((pca_df['raw_source_domain'] == 'final').sum())}`",
        f"- `added` 工况数：`{int((pca_df['raw_source_domain'] == 'added').sum())}`",
        f"- `added2` 工况数：`{int((pca_df['raw_source_domain'] == 'added2').sum())}`",
        f"- `competition_test` 样本数：`{int((pca_df['raw_source_domain'] == 'competition_test').sum())}`",
        f"- PCA explained variance：`PC1={pca.explained_variance_ratio_[0]:.2%}`, `PC2={pca.explained_variance_ratio_[1]:.2%}`",
        "",
        "## 四组测试输入坐标",
        "",
    ]
    for _, row in competition_df.iterrows():
        lines.append(
            f"- `{row['plot_label']}`: "
            f"`file={row['file_name']}`, `rpm={float(row['rpm']):.4f}`, "
            f"`pca1={float(row['pca1']):.4f}`, `pca2={float(row['pca2']):.4f}`"
        )

    if len(competition_df) >= 2:
        lines.extend(["", "## 重合说明", ""])
        overlap_check_df = competition_df.copy()
        overlap_check_df["pca1_rounded"] = overlap_check_df["pca1"].round(6)
        overlap_check_df["pca2_rounded"] = overlap_check_df["pca2"].round(6)
        dup_groups = (
            overlap_check_df.groupby(["file_name", "pca1_rounded", "pca2_rounded"], dropna=False)
            .agg(plot_labels=("plot_label", lambda x: ",".join(map(str, x))))
            .reset_index()
        )
        overlap_df = dup_groups.loc[dup_groups["plot_labels"].str.contains(",")].copy()
        if overlap_df.empty:
            lines.append("- 当前四组测试输入没有完全重合的 PCA 坐标。")
        else:
            for _, row in overlap_df.iterrows():
                lines.append(
                    f"- `{row['file_name']}` 对应的 `{row['plot_labels']}` 在当前 embedding PCA 下坐标完全重合。"
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
