from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "057_embedding_space_diagnosis"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY053_CKPT_DIR = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "models" / "checkpoints"
TRY053_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "053_support_window_residual_quickcheck" / "run_support_window_residual_quickcheck.py"
WINDOW_LABELS = ("2s", "8s")
TOP_K = 4
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一 embedding 空间诊断。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models" / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    try053 = load_try053_module()

    final_records_all = sorted(try053.scan_dataset_records(), key=lambda record: record.case_id)
    final_records_labeled = [record for record in final_records_all if record.is_labeled]
    added_records = sorted(try053.load_added_records(), key=lambda record: record.case_id)
    train_records = sorted([*final_records_labeled, *added_records], key=lambda record: record.case_id)
    export_records = sorted([*final_records_all, *added_records], key=lambda record: record.case_id)

    domain_by_case_id = {}
    for record in final_records_all:
        domain_by_case_id[record.case_id] = "final_labeled" if record.is_labeled else "final_unlabeled"
    for record in added_records:
        domain_by_case_id[record.case_id] = "added"

    common_signal_columns = try053.get_common_signal_columns(export_records)
    cleaned_signal_frames = {
        record.case_id: try053.load_clean_signal_frame(record, common_signal_columns)
        for record in export_records
    }

    per_window: dict[str, dict[str, object]] = {}
    for order, window_label in enumerate(WINDOW_LABELS, start=1):
        per_window[window_label] = load_or_train_unified_window_embeddings(
            try053=try053,
            train_records=train_records,
            export_records=export_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=try053.WINDOW_CONFIGS[window_label],
            window_label=window_label,
            seed=args.random_seed + order * 1000,
            read_checkpoint_dir=TRY053_CKPT_DIR,
            write_checkpoint_dir=model_dir,
            force_retrain=args.force_retrain,
        )

    embedding_case_df = build_embedding_case_table(export_records, per_window, domain_by_case_id)
    scaled_matrix, pca_df, pca_model, order_df = build_pca_table(embedding_case_df)
    distance_df = build_pairwise_distance_table(embedding_case_df, scaled_matrix)
    knn_df = build_knn_table(embedding_case_df, scaled_matrix, top_k=args.top_k)
    hubness_df = build_hubness_table(embedding_case_df, knn_df)

    embedding_case_df.to_csv(output_dir / "embedding_case_table.csv", index=False, encoding="utf-8-sig")
    pca_df.to_csv(output_dir / "embedding_pca_coords.csv", index=False, encoding="utf-8-sig")
    distance_df.to_csv(output_dir / "pairwise_distance_matrix.csv", index=False, encoding="utf-8-sig")
    knn_df.to_csv(output_dir / "knn_neighbors.csv", index=False, encoding="utf-8-sig")
    hubness_df.to_csv(output_dir / "hubness_counts.csv", index=False, encoding="utf-8-sig")

    create_pca_by_domain_plot(pca_df, plot_dir / "pca_by_domain.png")
    create_pca_by_wind_speed_plot(pca_df, plot_dir / "pca_by_wind_speed.png")
    create_pca_top1_edges_plot(pca_df, knn_df, plot_dir / "pca_top1_edges.png")
    create_pairwise_distance_heatmap(distance_df, order_df["case_id"].tolist(), plot_dir / "pairwise_distance_heatmap.png")
    create_hubness_bar_plot(hubness_df, plot_dir / "hubness_bar.png")
    write_summary_markdown(
        embedding_case_df=embedding_case_df,
        pca_df=pca_df,
        pca_model=pca_model,
        knn_df=knn_df,
        hubness_df=hubness_df,
        output_path=output_dir / "summary.md",
        top_k=args.top_k,
    )

    print("057 embedding 空间诊断已完成。")
    print(f"输出目录: {output_dir}")


def load_try053_module():
    spec = importlib.util.spec_from_file_location("try053_support_module", TRY053_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 053 脚本: {TRY053_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_or_train_unified_window_embeddings(
    *,
    try053,
    train_records,
    export_records,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config,
    window_label: str,
    seed: int,
    read_checkpoint_dir: Path,
    write_checkpoint_dir: Path,
    force_retrain: bool,
) -> dict[str, object]:
    train_dataset = try053.build_raw_window_dataset(
        train_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in train_records},
        window_config,
    )
    export_dataset = try053.build_raw_window_dataset(
        export_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in export_records},
        window_config,
    )

    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_export = export_dataset.windows

    read_base = read_checkpoint_dir / f"unified_all_{window_label}"
    write_base = write_checkpoint_dir / f"unified_all_{window_label}"
    model = try053.TinyTCNEncoderRegressor(in_channels=X_train.shape[1])

    existing_paths = {
        "ckpt": read_base.with_suffix(".pt"),
        "norm": read_checkpoint_dir / f"unified_all_{window_label}_norm.npz",
        "meta": read_checkpoint_dir / f"unified_all_{window_label}.json",
    }
    cache_paths = {
        "ckpt": write_base.with_suffix(".pt"),
        "norm": write_checkpoint_dir / f"unified_all_{window_label}_norm.npz",
        "meta": write_checkpoint_dir / f"unified_all_{window_label}.json",
    }

    if all(path.exists() for path in existing_paths.values()) and not force_retrain:
        state = torch.load(existing_paths["ckpt"], map_location="cpu")
        model.load_state_dict(state)
        norm = np.load(existing_paths["norm"])
        mean = norm["mean"]
        std = norm["std"]
    elif all(path.exists() for path in cache_paths.values()) and not force_retrain:
        state = torch.load(cache_paths["ckpt"], map_location="cpu")
        model.load_state_dict(state)
        norm = np.load(cache_paths["norm"])
        mean = norm["mean"]
        std = norm["std"]
    else:
        X_train_norm, _, mean, std = try053.normalize_windows_by_channel(X_train, X_train)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model.to(torch.device("cpu"))
        try053.train_model(model, X_train_norm, y_train, try053.TrainConfig(), torch.device("cpu"))
        torch.save(model.state_dict(), cache_paths["ckpt"])
        np.savez(cache_paths["norm"], mean=mean, std=std)
        cache_paths["meta"].write_text(
            json.dumps(
                {
                    "window_label": window_label,
                    "seed": seed,
                    "train_case_ids": [int(record.case_id) for record in train_records],
                    "export_case_ids": [int(record.case_id) for record in export_records],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = model.cpu()

    X_export_norm = ((X_export - mean) / std).astype(np.float32)
    with torch.no_grad():
        export_tensor = torch.from_numpy(X_export_norm).float()
        export_embedding = model.encode(export_tensor).cpu().numpy()

    return {
        "export_meta_df": export_dataset.meta_df.copy().reset_index(drop=True),
        "export_embedding": export_embedding,
    }


def build_embedding_case_table(
    export_records,
    per_window: dict[str, dict[str, object]],
    domain_by_case_id: dict[int, str],
) -> pd.DataFrame:
    record_rows = pd.DataFrame(
        [
            {
                "case_id": int(record.case_id),
                "file_name": str(record.file_name),
                "domain": domain_by_case_id[int(record.case_id)],
                "wind_speed": np.nan if record.wind_speed is None else float(record.wind_speed),
                "rpm": np.nan if record.rpm is None else float(record.rpm),
                "is_labeled": bool(record.is_labeled),
            }
            for record in export_records
        ]
    ).sort_values("case_id").reset_index(drop=True)

    result = record_rows.copy()
    for window_label in WINDOW_LABELS:
        embedding_df = aggregate_case_embeddings(
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
    concat_matrix = np.vstack(result["embedding_concat"].to_numpy())
    for index in range(concat_matrix.shape[1]):
        result[f"embedding_{index + 1}"] = concat_matrix[:, index]
    return result.drop(columns=["embedding_2s", "embedding_8s", "embedding_concat"]).reset_index(drop=True)


def aggregate_case_embeddings(meta_df: pd.DataFrame, window_embeddings: np.ndarray, column_name: str) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []
    for case_id, block in meta_df.groupby("case_id", sort=False):
        indices = block.index.to_numpy(dtype=int, copy=False)
        grouped_rows.append(
            {
                "case_id": int(case_id),
                column_name: window_embeddings[indices].mean(axis=0).astype(float),
            }
        )
    return pd.DataFrame(grouped_rows)


def build_pca_table(embedding_case_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, PCA, pd.DataFrame]:
    feature_columns = [column for column in embedding_case_df.columns if column.startswith("embedding_")]
    matrix = embedding_case_df[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_matrix)
    pca_df = embedding_case_df[
        ["case_id", "file_name", "domain", "wind_speed", "rpm", "is_labeled"]
    ].copy()
    pca_df["pca1"] = coords[:, 0]
    pca_df["pca2"] = coords[:, 1]
    pca_df["pca_radius"] = np.sqrt(np.square(coords[:, 0]) + np.square(coords[:, 1]))
    order_df = pca_df.sort_values(["domain", "pca1", "case_id"]).reset_index(drop=True)
    return scaled_matrix, pca_df, pca, order_df


def build_pairwise_distance_table(embedding_case_df: pd.DataFrame, scaled_matrix: np.ndarray) -> pd.DataFrame:
    case_ids = embedding_case_df["case_id"].to_numpy(dtype=int, copy=False)
    diff = scaled_matrix[:, None, :] - scaled_matrix[None, :, :]
    distances = np.sqrt(np.sum(np.square(diff), axis=2))
    result = pd.DataFrame(distances, index=case_ids, columns=case_ids)
    result.index.name = "case_id"
    return result.reset_index()


def build_knn_table(embedding_case_df: pd.DataFrame, scaled_matrix: np.ndarray, *, top_k: int) -> pd.DataFrame:
    case_ids = embedding_case_df["case_id"].to_numpy(dtype=int, copy=False)
    domains = embedding_case_df["domain"].tolist()
    winds = embedding_case_df["wind_speed"].to_numpy(dtype=float, copy=False)
    rpms = embedding_case_df["rpm"].to_numpy(dtype=float, copy=False)

    diff = scaled_matrix[:, None, :] - scaled_matrix[None, :, :]
    distances = np.sqrt(np.sum(np.square(diff), axis=2))
    rows: list[dict[str, object]] = []
    for row_idx, case_id in enumerate(case_ids):
        order = np.argsort(distances[row_idx])
        order = order[order != row_idx][: min(top_k, len(case_ids) - 1)]
        weights = 1.0 / np.maximum(distances[row_idx, order], EPS)
        weights = weights / weights.sum()
        for rank, (col_idx, weight) in enumerate(zip(order, weights, strict=True), start=1):
            rows.append(
                {
                    "case_id": int(case_id),
                    "domain": domains[row_idx],
                    "neighbor_rank": rank,
                    "neighbor_case_id": int(case_ids[col_idx]),
                    "neighbor_domain": domains[col_idx],
                    "distance": float(distances[row_idx, col_idx]),
                    "weight": float(weight),
                    "same_domain": bool(domains[row_idx] == domains[col_idx]),
                    "wind_speed": float(winds[row_idx]) if not np.isnan(winds[row_idx]) else np.nan,
                    "neighbor_wind_speed": float(winds[col_idx]) if not np.isnan(winds[col_idx]) else np.nan,
                    "rpm": float(rpms[row_idx]) if not np.isnan(rpms[row_idx]) else np.nan,
                    "neighbor_rpm": float(rpms[col_idx]) if not np.isnan(rpms[col_idx]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_hubness_table(embedding_case_df: pd.DataFrame, knn_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        knn_df.groupby("neighbor_case_id")
        .size()
        .rename("selected_count")
        .reset_index()
        .rename(columns={"neighbor_case_id": "case_id"})
    )
    top1_counts = (
        knn_df.loc[knn_df["neighbor_rank"] == 1]
        .groupby("neighbor_case_id")
        .size()
        .rename("top1_selected_count")
        .reset_index()
        .rename(columns={"neighbor_case_id": "case_id"})
    )
    result = embedding_case_df[["case_id", "file_name", "domain", "wind_speed", "rpm"]].merge(
        counts, on="case_id", how="left"
    ).merge(
        top1_counts, on="case_id", how="left"
    )
    result["selected_count"] = result["selected_count"].fillna(0).astype(int)
    result["top1_selected_count"] = result["top1_selected_count"].fillna(0).astype(int)
    return result.sort_values(["selected_count", "top1_selected_count", "case_id"], ascending=[False, False, True]).reset_index(drop=True)


def create_pca_by_domain_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    style_map = {
        "final_labeled": ("tab:blue", "o"),
        "final_unlabeled": ("gray", "s"),
        "added": ("tab:orange", "^"),
    }
    for domain_name, block in pca_df.groupby("domain", sort=False):
        color, marker = style_map.get(domain_name, ("black", "o"))
        ax.scatter(block["pca1"], block["pca2"], s=90, c=color, marker=marker, label=domain_name)
    for _, row in pca_df.iterrows():
        ax.annotate(str(int(row["case_id"])), (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Unified embedding PCA colored by domain")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_pca_by_wind_speed_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    labeled = pca_df.loc[pca_df["is_labeled"]].copy()
    unlabeled = pca_df.loc[~pca_df["is_labeled"]].copy()
    scatter = ax.scatter(
        labeled["pca1"],
        labeled["pca2"],
        c=labeled["wind_speed"],
        cmap="viridis",
        s=95,
    )
    if not unlabeled.empty:
        ax.scatter(unlabeled["pca1"], unlabeled["pca2"], c="gray", marker="s", s=95, label="unlabeled")
    for _, row in pca_df.iterrows():
        ax.annotate(str(int(row["case_id"])), (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Unified embedding PCA colored by wind speed")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if not unlabeled.empty:
        ax.legend()
    fig.colorbar(scatter, ax=ax, label="wind_speed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_pca_top1_edges_plot(pca_df: pd.DataFrame, knn_df: pd.DataFrame, output_path: Path) -> None:
    coord_by_case_id = pca_df.set_index("case_id")[["pca1", "pca2", "domain"]].to_dict("index")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    style_map = {
        "final_labeled": ("tab:blue", "o"),
        "final_unlabeled": ("gray", "s"),
        "added": ("tab:orange", "^"),
    }
    for domain_name, block in pca_df.groupby("domain", sort=False):
        color, marker = style_map.get(domain_name, ("black", "o"))
        ax.scatter(block["pca1"], block["pca2"], s=90, c=color, marker=marker, label=domain_name, zorder=3)

    top1_df = knn_df.loc[knn_df["neighbor_rank"] == 1].copy()
    for _, row in top1_df.iterrows():
        source = coord_by_case_id[int(row["case_id"])]
        target = coord_by_case_id[int(row["neighbor_case_id"])]
        line_color = "tab:orange" if source["domain"] == "added" or target["domain"] == "added" else "lightgray"
        ax.plot([source["pca1"], target["pca1"]], [source["pca2"], target["pca2"]], color=line_color, linewidth=1.2, alpha=0.8, zorder=1)

    for _, row in pca_df.iterrows():
        ax.annotate(str(int(row["case_id"])), (row["pca1"], row["pca2"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Unified embedding PCA with top-1 neighbor edges")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_pairwise_distance_heatmap(distance_df: pd.DataFrame, ordered_case_ids: list[int], output_path: Path) -> None:
    matrix = distance_df.set_index("case_id").loc[ordered_case_ids, ordered_case_ids].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(ordered_case_ids)))
    ax.set_xticklabels([str(case_id) for case_id in ordered_case_ids], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(ordered_case_ids)))
    ax.set_yticklabels([str(case_id) for case_id in ordered_case_ids], fontsize=7)
    ax.set_title("Pairwise embedding distance heatmap")
    ax.set_xlabel("case_id")
    ax.set_ylabel("case_id")
    fig.colorbar(im, ax=ax, label="euclidean distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_hubness_bar_plot(hubness_df: pd.DataFrame, output_path: Path) -> None:
    top_df = hubness_df.head(12).copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["tab:orange" if domain == "added" else ("gray" if domain == "final_unlabeled" else "tab:blue") for domain in top_df["domain"]]
    ax.bar([str(int(case_id)) for case_id in top_df["case_id"]], top_df["selected_count"], color=colors)
    for idx, row in top_df.reset_index(drop=True).iterrows():
        ax.text(idx, row["selected_count"] + 0.1, f"top1={int(row['top1_selected_count'])}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Hubness: most frequently retrieved cases")
    ax.set_xlabel("case_id")
    ax.set_ylabel("selected_count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    *,
    embedding_case_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    pca_model: PCA,
    knn_df: pd.DataFrame,
    hubness_df: pd.DataFrame,
    output_path: Path,
    top_k: int,
) -> None:
    overall_same_domain = float(knn_df["same_domain"].mean())
    added_same_domain = float(knn_df.loc[knn_df["domain"] == "added", "same_domain"].mean())
    final_same_domain = float(knn_df.loc[knn_df["domain"] == "final_labeled", "same_domain"].mean())
    top_hubs = hubness_df.head(5)

    lines = [
        "# unified embedding space diagnosis",
        "",
        f"- 导出工况数：`{len(embedding_case_df)}`",
        f"- 训练工况数：`{int(embedding_case_df['is_labeled'].sum())}`",
        f"- kNN 口径：`top-{top_k}`",
        f"- PCA explained variance：`PC1={pca_model.explained_variance_ratio_[0]:.2%}`, `PC2={pca_model.explained_variance_ratio_[1]:.2%}`",
        f"- overall same-domain neighbor rate：`{overall_same_domain:.2%}`",
        f"- final_labeled same-domain neighbor rate：`{final_same_domain:.2%}`",
        f"- added same-domain neighbor rate：`{added_same_domain:.2%}`",
        "",
        "## Added 邻居摘要",
        "",
    ]

    for case_id in sorted(embedding_case_df.loc[embedding_case_df["domain"] == "added", "case_id"].tolist()):
        block = knn_df.loc[knn_df["case_id"] == case_id].sort_values("neighbor_rank")
        neighbor_text = ", ".join(
            f"{int(row['neighbor_case_id'])}({row['distance']:.2f})"
            for _, row in block.iterrows()
        )
        lines.append(f"- `工况{case_id}` -> {neighbor_text}")

    lines.extend(["", "## Top Hub Cases", ""])
    for _, row in top_hubs.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: domain=`{row['domain']}`, selected_count=`{int(row['selected_count'])}`, top1_selected_count=`{int(row['top1_selected_count'])}`"
        )

    lines.extend(["", "## 说明", ""])
    lines.append("- 当前图和邻域关系来自统一训练的一组 `2s+8s` encoder，不再混用 holdout fold 坐标系。")
    lines.append("- 本文档只描述空间结构与检索行为，不直接下最终预测优劣结论。")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
