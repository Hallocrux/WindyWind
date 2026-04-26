from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from reuse_embedding_domain_common import (
    REPO_ROOT,
    TRY057_CKPT_DIR,
    WINDOW_LABELS,
    build_embedding_case_table,
    build_record_table,
    build_cleaned_signal_frames,
    get_embedding_columns,
    load_fixed_window_embeddings,
    load_previous_057_case_table,
    load_source_catalog,
    load_try053_module,
)

TRY_NAME = "066_reuse_embedding_domain_split"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
TOP_K = 4
MIN_DOMAIN_SIZE = 4
K_CANDIDATES = (4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用旧 embedding 完成新域划分。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--embedding-checkpoint-root", type=Path, default=TRY057_CKPT_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-domain-size", type=int, default=MIN_DOMAIN_SIZE)
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
        checkpoint_root=args.embedding_checkpoint_root,
    )
    embedding_case_df = build_embedding_case_table(record_df, fixed_window_embeddings)
    embedding_columns = get_embedding_columns(embedding_case_df)

    validation_df = build_embedding_reuse_validation(embedding_case_df, embedding_columns)
    labeled_df = embedding_case_df.loc[embedding_case_df["is_labeled"]].copy().reset_index(drop=True)
    clustering_input_df = labeled_df.loc[labeled_df["case_id"] != 2].copy().reset_index(drop=True)
    scaled_matrix, scaler = scale_embeddings(clustering_input_df, embedding_columns)
    cluster_selection_df, selected = select_clustering_solution(
        clustering_input_df,
        scaled_matrix,
        min_domain_size=args.min_domain_size,
    )
    assignment_df = build_domain_assignment(
        embedding_case_df=embedding_case_df,
        clustering_input_df=clustering_input_df,
        scaled_matrix=scaled_matrix,
        scaler=scaler,
        selected_solution=selected,
        embedding_columns=embedding_columns,
        min_domain_size=args.min_domain_size,
    )
    summary_df = build_domain_summary(assignment_df)
    knn_df = build_knn_neighbors(assignment_df, embedding_columns, top_k=args.top_k)

    pca_df, explained_variance = build_pca_table(assignment_df, embedding_columns)
    create_pca_by_source_plot(pca_df, plot_dir / "pca_by_source.png")
    create_pca_by_domain_plot(pca_df, plot_dir / "pca_by_learned_domain.png")

    embedding_case_df.to_csv(output_dir / "embedding_case_table.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(output_dir / "embedding_reuse_validation.csv", index=False, encoding="utf-8-sig")
    cluster_selection_df.to_csv(output_dir / "cluster_selection_report.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(output_dir / "domain_assignment.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "domain_summary.csv", index=False, encoding="utf-8-sig")
    knn_df.to_csv(output_dir / "knn_neighbors.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_dir / "summary.md",
        cluster_selection_df=cluster_selection_df,
        assignment_df=assignment_df,
        summary_df=summary_df,
        validation_df=validation_df,
        explained_variance=explained_variance,
    )

    print("066 reuse embedding domain split 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        "选定域划分: "
        f"K={int(selected['selected_k'])}, "
        f"assignment_mode={selected['assignment_strategy']}, "
        f"silhouette={selected['selected_silhouette_score']:.4f}"
    )


def build_embedding_reuse_validation(embedding_case_df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    previous_df = load_previous_057_case_table()
    previous_columns = [column for column in previous_df.columns if column.startswith("embedding_")]
    overlap_case_ids = sorted(set(previous_df["case_id"].tolist()) & set(embedding_case_df["case_id"].tolist()))
    merged = (
        embedding_case_df.loc[embedding_case_df["case_id"].isin(overlap_case_ids), ["case_id", *embedding_columns]]
        .merge(previous_df.loc[previous_df["case_id"].isin(overlap_case_ids), ["case_id", *previous_columns]], on="case_id", suffixes=("_new", "_old"))
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        new_values = row[[f"{column}_new" for column in embedding_columns]].to_numpy(dtype=float, copy=False)
        old_values = row[[f"{column}_old" for column in previous_columns]].to_numpy(dtype=float, copy=False)
        rows.append(
            {
                "case_id": int(row["case_id"]),
                "new_dim": int(len(new_values)),
                "old_dim": int(len(old_values)),
                "dimension_match": int(len(new_values) == len(old_values)),
                "max_abs_diff": float(np.max(np.abs(new_values - old_values))),
                "mean_abs_diff": float(np.mean(np.abs(new_values - old_values))),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def scale_embeddings(embedding_df: pd.DataFrame, embedding_columns: list[str]) -> tuple[np.ndarray, StandardScaler]:
    matrix = embedding_df[embedding_columns].to_numpy(dtype=float, copy=False)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def select_clustering_solution(
    embedding_df: pd.DataFrame,
    scaled_matrix: np.ndarray,
    *,
    min_domain_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_solutions: list[dict[str, object]] = []
    for k in K_CANDIDATES:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled_matrix)
        counts = pd.Series(labels).value_counts().sort_index()
        silhouette = float(silhouette_score(scaled_matrix, labels))
        rows.append(
            {
                "k": int(k),
                "silhouette_score": silhouette,
                "min_cluster_size": int(counts.min()),
                "is_feasible": int(counts.min() >= min_domain_size),
                "cluster_sizes": ",".join(str(int(value)) for value in counts.tolist()),
            }
        )
        raw_solutions.append(
            {
                "k": int(k),
                "labels": labels,
                "silhouette_score": silhouette,
                "counts": counts,
            }
        )

    report_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    feasible = [solution for solution in raw_solutions if int(solution["counts"].min()) >= min_domain_size]
    if feasible:
        feasible = sorted(feasible, key=lambda item: (-float(item["silhouette_score"]), int(item["k"])))
        best = feasible[0]
        if len(feasible) > 1:
            runner_up = feasible[1]
            if abs(float(best["silhouette_score"]) - float(runner_up["silhouette_score"])) < 0.02:
                best = min(feasible[:2], key=lambda item: int(item["k"]))
        selected = {
            "selected_k": int(best["k"]),
            "labels_before_merge": best["labels"],
            "selected_silhouette_score": float(best["silhouette_score"]),
            "assignment_strategy": "auto_cluster",
        }
    else:
        best = max(raw_solutions, key=lambda item: (float(item["silhouette_score"]), -int(item["k"])))
        selected = {
            "selected_k": int(best["k"]),
            "labels_before_merge": best["labels"],
            "selected_silhouette_score": float(best["silhouette_score"]),
            "assignment_strategy": "manual_merge_required",
        }

    report_df["selected"] = (report_df["k"] == int(selected["selected_k"])).astype(int)
    return report_df, selected


def build_domain_assignment(
    *,
    embedding_case_df: pd.DataFrame,
    clustering_input_df: pd.DataFrame,
    scaled_matrix: np.ndarray,
    scaler: StandardScaler,
    selected_solution: dict[str, object],
    embedding_columns: list[str],
    min_domain_size: int,
) -> pd.DataFrame:
    working_df = clustering_input_df.copy().reset_index(drop=True)
    raw_labels = np.asarray(selected_solution["labels_before_merge"], dtype=int)
    merged_labels, assignment_modes = merge_small_clusters(raw_labels, scaled_matrix, min_cluster_size=min_domain_size)
    if any(mode == "manual_merge" for mode in assignment_modes):
        selected_solution["assignment_strategy"] = "manual_merge"

    working_df["cluster_before_merge"] = raw_labels
    working_df["cluster_after_merge"] = merged_labels
    working_df["assignment_mode"] = assignment_modes

    cluster_meta = build_cluster_meta(working_df, scaled_matrix)
    working_df = working_df.merge(cluster_meta, left_on="cluster_after_merge", right_on="cluster_after_merge", how="left")

    centers = {
        int(row["cluster_after_merge"]): np.asarray(row["scaled_centroid"], dtype=float)
        for _, row in cluster_meta.iterrows()
    }
    silhouette = silhouette_samples(scaled_matrix, merged_labels)
    working_df["silhouette_sample"] = silhouette
    working_df["distance_to_centroid"] = [
        float(np.linalg.norm(scaled_matrix[idx] - centers[int(label)]))
        for idx, label in enumerate(merged_labels)
    ]

    unlabeled_df = embedding_case_df.loc[~embedding_case_df["is_labeled"]].copy().reset_index(drop=True)
    if not unlabeled_df.empty:
        unlabeled_matrix = scaler.transform(unlabeled_df[embedding_columns].to_numpy(dtype=float, copy=False))
        domain_rows: list[dict[str, object]] = []
        for row_idx, (_, row) in enumerate(unlabeled_df.iterrows()):
            distances = {
                cluster_id: float(np.linalg.norm(unlabeled_matrix[row_idx] - centroid))
                for cluster_id, centroid in centers.items()
            }
            best_cluster = min(distances, key=distances.get)
            learned_name = str(cluster_meta.loc[cluster_meta["cluster_after_merge"] == best_cluster, "learned_domain_name"].iloc[0])
            domain_rows.append(
                {
                    "case_id": int(row["case_id"]),
                    "raw_source_domain": str(row["raw_source_domain"]),
                    "is_labeled": bool(row["is_labeled"]),
                    "learned_domain_id": int(best_cluster),
                    "learned_domain_name": learned_name,
                    "assignment_mode": "nearest_centroid_unlabeled",
                    "distance_to_centroid": float(distances[best_cluster]),
                    "silhouette_sample": np.nan,
                }
            )
        unlabeled_assignment_df = pd.DataFrame(domain_rows)
    else:
        unlabeled_assignment_df = pd.DataFrame(
            columns=[
                "case_id",
                "raw_source_domain",
                "is_labeled",
                "learned_domain_id",
                "learned_domain_name",
                "assignment_mode",
                "distance_to_centroid",
                "silhouette_sample",
            ]
        )

    labeled_assignment_df = working_df[
        [
            "case_id",
            "raw_source_domain",
            "is_labeled",
            "cluster_after_merge",
            "learned_domain_name",
            "assignment_mode",
            "distance_to_centroid",
            "silhouette_sample",
        ]
    ].rename(columns={"cluster_after_merge": "learned_domain_id"})

    assignment_df = (
        pd.concat([labeled_assignment_df, unlabeled_assignment_df], ignore_index=True)
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    return assignment_df.merge(embedding_case_df[["case_id", *embedding_columns]], on="case_id", how="left")


def merge_small_clusters(labels: np.ndarray, scaled_matrix: np.ndarray, *, min_cluster_size: int) -> tuple[np.ndarray, list[str]]:
    merged_labels = labels.copy()
    assignment_modes = ["auto_cluster"] * len(labels)
    while True:
        counts = pd.Series(merged_labels).value_counts().sort_index()
        small_clusters = [int(cluster_id) for cluster_id, count in counts.items() if int(count) < min_cluster_size]
        stable_clusters = [int(cluster_id) for cluster_id, count in counts.items() if int(count) >= min_cluster_size]
        if not small_clusters or not stable_clusters:
            break
        centroids = {
            cluster_id: scaled_matrix[merged_labels == cluster_id].mean(axis=0)
            for cluster_id in counts.index.tolist()
        }
        for small_cluster in small_clusters:
            target_cluster = min(
                stable_clusters,
                key=lambda cluster_id: float(np.linalg.norm(centroids[small_cluster] - centroids[cluster_id])),
            )
            for idx in np.flatnonzero(merged_labels == small_cluster):
                merged_labels[idx] = target_cluster
                assignment_modes[idx] = "manual_merge"
    return renumber_labels(merged_labels), assignment_modes


def renumber_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(set(int(value) for value in labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in labels.tolist()], dtype=int)


def build_cluster_meta(working_df: pd.DataFrame, scaled_matrix: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(working_df["cluster_after_merge"].unique()):
        indices = working_df.index[working_df["cluster_after_merge"] == cluster_id].to_numpy(dtype=int, copy=False)
        block = working_df.loc[indices]
        rows.append(
            {
                "cluster_after_merge": int(cluster_id),
                "learned_domain_name": f"domain_{int(cluster_id)}",
                "scaled_centroid": scaled_matrix[indices].mean(axis=0).astype(float),
                "cluster_case_count": int(len(indices)),
                "cluster_case_ids": ",".join(str(int(case_id)) for case_id in block["case_id"].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_after_merge").reset_index(drop=True)


def build_domain_summary(assignment_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for learned_domain_id, block in assignment_df.groupby("learned_domain_id", sort=True):
        raw_counts = block["raw_source_domain"].value_counts().sort_index()
        rows.append(
            {
                "learned_domain_id": int(learned_domain_id),
                "learned_domain_name": str(block["learned_domain_name"].iloc[0]),
                "case_count": int(len(block)),
                "labeled_case_count": int(block["is_labeled"].sum()),
                "case_ids": ",".join(str(int(case_id)) for case_id in block["case_id"].tolist()),
                "raw_source_mix": ",".join(f"{domain}:{int(count)}" for domain, count in raw_counts.items()),
                "assignment_mode_breakdown": ",".join(
                    f"{mode}:{int(count)}" for mode, count in block["assignment_mode"].value_counts().sort_index().items()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("learned_domain_id").reset_index(drop=True)


def build_knn_neighbors(assignment_df: pd.DataFrame, embedding_columns: list[str], *, top_k: int) -> pd.DataFrame:
    matrix = assignment_df.loc[assignment_df["is_labeled"]].copy().reset_index(drop=True)
    scaled = StandardScaler().fit_transform(matrix[embedding_columns].to_numpy(dtype=float, copy=False))
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(matrix)), metric="euclidean")
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled)
    rows: list[dict[str, object]] = []
    for row_idx, case_id in enumerate(matrix["case_id"].tolist()):
        rank = 0
        for neighbor_pos, neighbor_idx in enumerate(indices[row_idx].tolist()):
            if neighbor_idx == row_idx:
                continue
            rank += 1
            if rank > top_k:
                break
            rows.append(
                {
                    "case_id": int(case_id),
                    "neighbor_rank": int(rank),
                    "neighbor_case_id": int(matrix.iloc[neighbor_idx]["case_id"]),
                    "distance": float(distances[row_idx][neighbor_pos]),
                    "same_learned_domain": int(
                        matrix.iloc[row_idx]["learned_domain_id"] == matrix.iloc[neighbor_idx]["learned_domain_id"]
                    ),
                    "same_raw_source_domain": int(
                        matrix.iloc[row_idx]["raw_source_domain"] == matrix.iloc[neighbor_idx]["raw_source_domain"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "neighbor_rank"]).reset_index(drop=True)


def build_pca_table(assignment_df: pd.DataFrame, embedding_columns: list[str]) -> tuple[pd.DataFrame, tuple[float, float]]:
    matrix = assignment_df[embedding_columns].to_numpy(dtype=float, copy=False)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    pca_df = assignment_df[["case_id", "raw_source_domain", "learned_domain_name", "is_labeled"]].copy()
    pca_df["pca1"] = coords[:, 0]
    pca_df["pca2"] = coords[:, 1]
    explained = tuple(float(value) for value in pca.explained_variance_ratio_)
    return pca_df, explained


def create_pca_by_source_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {"final": "#386cb0", "added": "#fdb462", "added2": "#ef3b2c"}
    for raw_source_domain, block in pca_df.groupby("raw_source_domain", sort=True):
        ax.scatter(block["pca1"], block["pca2"], s=80, label=raw_source_domain, alpha=0.85, color=color_map.get(raw_source_domain))
    for _, row in pca_df.iterrows():
        ax.text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=8, ha="center", va="center")
    ax.set_title("PCA by raw source domain")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_pca_by_domain_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10")
    domains = sorted(pca_df["learned_domain_name"].unique())
    for index, domain_name in enumerate(domains):
        block = pca_df.loc[pca_df["learned_domain_name"] == domain_name]
        ax.scatter(block["pca1"], block["pca2"], s=80, label=domain_name, alpha=0.85, color=cmap(index))
    for _, row in pca_df.iterrows():
        ax.text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=8, ha="center", va="center")
    ax.set_title("PCA by learned domain")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    *,
    cluster_selection_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    explained_variance: tuple[float, float],
) -> None:
    selected_row = cluster_selection_df.loc[cluster_selection_df["selected"] == 1].iloc[0]
    lines = [
        "# reuse embedding domain split",
        "",
        "## 选择结果",
        "",
        f"- 选定 `K`：`{int(selected_row['k'])}`",
        f"- silhouette：`{float(selected_row['silhouette_score']):.4f}`",
        f"- 最小簇大小：`{int(selected_row['min_cluster_size'])}`",
        f"- PCA explained variance：`PC1={explained_variance[0]:.4f}`, `PC2={explained_variance[1]:.4f}`",
        "",
        "## 旧 embedding 复用校验",
        "",
        f"- 重叠工况数：`{int(len(validation_df))}`",
        f"- 维度全部一致：`{int(validation_df['dimension_match'].all())}`",
        f"- 最大 `max_abs_diff`：`{float(validation_df['max_abs_diff'].max()):.8f}`",
        "",
        "## learned domain 摘要",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['learned_domain_name']}`: case_count=`{int(row['case_count'])}`, "
            f"labeled_case_count=`{int(row['labeled_case_count'])}`, "
            f"case_ids=`{row['case_ids']}`, raw_source_mix=`{row['raw_source_mix']}`"
        )
    lines.extend(["", "## 特殊样本", ""])
    case2 = assignment_df.loc[assignment_df["case_id"] == 2]
    if not case2.empty:
        row = case2.iloc[0]
        lines.append(
            f"- `工况2`: learned_domain=`{row['learned_domain_name']}`, "
            f"assignment_mode=`{row['assignment_mode']}`, "
            f"distance_to_centroid=`{float(row['distance_to_centroid']):.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
