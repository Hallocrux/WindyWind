from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "058_prototype_head_ablation_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY052_CASE_PATH = REPO_ROOT / "outputs" / "try" / "052_tcn_embedding_window_signal_quickcheck" / "case_level_predictions.csv"
TRY053_CASE_PATH = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "case_level_predictions.csv"
TRY056_CASE_PATH = REPO_ROOT / "outputs" / "try" / "056_embedding_topk_local_prototype_fusion" / "case_level_predictions.csv"
TRY053_CKPT_DIR = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "models" / "checkpoints"
TRY053_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "053_support_window_residual_quickcheck" / "run_support_window_residual_quickcheck.py"
HOLDOUT_CASE_IDS = [1, 3, 17, 18, 21, 22, 23, 24]
WINDOW_LABELS = ("2s", "8s")
BASELINE_VARIANTS = [
    "rpm_knn4",
    "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
    "rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5",
    "rpm_knn4__plus__embedding_topk_prototype_ridge_w0.5",
]
VARIANT_DELTA_ONLY = "rpm_knn4__plus__delta_only_prototype_ridge_w0.5"
VARIANT_LOWRANK = "rpm_knn4__plus__lowrank_delta_prototype_ridge_w0.5"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
EPS = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    top_k: int = 4
    delta_pca_components: int = 6
    random_seed: int = 42
    force_retrain: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="prototype head ablation quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--delta-pca-components", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_cache_dir = model_dir / "encoder_checkpoints"
    encoder_cache_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        top_k=args.top_k,
        delta_pca_components=args.delta_pca_components,
        random_seed=args.random_seed,
        force_retrain=args.force_retrain,
    )
    try053 = load_try053_module()
    baseline_case_df = load_baseline_cases()

    final_records = [record for record in try053.scan_dataset_records() if record.is_labeled]
    added_records = try053.load_added_records()
    all_records = sorted([*final_records, *added_records], key=lambda record: record.case_id)
    record_by_case_id = {record.case_id: record for record in all_records}
    holdout_records = [record_by_case_id[case_id] for case_id in HOLDOUT_CASE_IDS]
    domain_by_case_id = {case_id: ("final_focus" if case_id < 21 else "added_focus") for case_id in HOLDOUT_CASE_IDS}

    common_signal_columns = try053.get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: try053.load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    delta_pca_rows: list[dict[str, object]] = []
    for holdout in holdout_records:
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        fold_case_rows, fold_neighbor_rows, fold_feature_rows, fold_delta_pca_rows = run_holdout_fold(
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            domain_name=domain_by_case_id[holdout.case_id],
            model_dir=model_dir,
            encoder_cache_dir=encoder_cache_dir,
            config=config,
        )
        case_rows.extend(fold_case_rows)
        neighbor_rows.extend(fold_neighbor_rows)
        feature_rows.extend(fold_feature_rows)
        delta_pca_rows.extend(fold_delta_pca_rows)

    new_case_df = pd.DataFrame(case_rows)
    neighbor_df = pd.DataFrame(neighbor_rows)
    feature_df = pd.DataFrame(feature_rows)
    delta_pca_df = pd.DataFrame(delta_pca_rows)
    merged_case_df = pd.concat([baseline_case_df, new_case_df], ignore_index=True)
    summary_df = build_summary_by_domain(merged_case_df)

    merged_case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(output_dir / "reference_neighbors.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_dir / "prototype_feature_table.csv", index=False, encoding="utf-8-sig")
    delta_pca_df.to_csv(output_dir / "delta_pca_feature_table.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, merged_case_df)

    best_focus = summary_df.loc[summary_df["domain"] == "focus_all"].iloc[0]
    print("058 prototype head ablation quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best focus_all: {best_focus['variant_name']} | case_mae={best_focus['case_mae']:.4f}")


def load_try053_module():
    spec = importlib.util.spec_from_file_location("try053_support_module_058", TRY053_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 053 脚本: {TRY053_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline_cases() -> pd.DataFrame:
    case_052 = pd.read_csv(TRY052_CASE_PATH, encoding="utf-8-sig")
    case_053 = pd.read_csv(TRY053_CASE_PATH, encoding="utf-8-sig")
    case_056 = pd.read_csv(TRY056_CASE_PATH, encoding="utf-8-sig")
    frames = [
        case_052.loc[case_052["variant_name"].isin(BASELINE_VARIANTS[:2])].copy(),
        case_053.loc[case_053["variant_name"] == BASELINE_VARIANTS[2]].copy(),
        case_056.loc[case_056["variant_name"] == BASELINE_VARIANTS[3]].copy(),
    ]
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def run_holdout_fold(
    *,
    try053,
    train_records,
    holdout,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    domain_name: str,
    model_dir: Path,
    encoder_cache_dir: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_case_df = build_case_table(train_records)
    train_case_df = add_rpm_oof_predictions(train_case_df, try053)
    holdout_base_pred = float(try053.predict_rpm_knn4_with_neighbors(train_records, float(holdout.rpm))[0])

    per_window: dict[str, dict[str, object]] = {}
    for order, window_label in enumerate(WINDOW_LABELS, start=1):
        window_config = try053.WINDOW_CONFIGS[window_label]
        per_window[window_label] = load_window_embeddings(
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            window_label=window_label,
            seed=config.random_seed + holdout.case_id * 100 + order,
            read_checkpoint_dir=TRY053_CKPT_DIR,
            write_checkpoint_dir=encoder_cache_dir,
            force_retrain=config.force_retrain,
        )

    train_case_df = attach_case_embeddings(train_case_df, per_window)
    holdout_row = build_holdout_case_row(holdout, holdout_base_pred, per_window)

    concat_matrix = np.vstack(train_case_df["embedding_concat"].to_numpy())
    embed_mean = concat_matrix.mean(axis=0, keepdims=True)
    embed_std = concat_matrix.std(axis=0, keepdims=True)
    embed_std = np.where(embed_std > 0, embed_std, 1.0)

    train_feature_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for _, target_row in train_case_df.iterrows():
        candidate_df = train_case_df.loc[train_case_df["case_id"] != int(target_row["case_id"])].reset_index(drop=True)
        feature_row, fold_neighbor_rows = build_case_feature_row(
            target_row=target_row,
            candidate_df=candidate_df,
            base_pred_column="rpm_pred_oof",
            residual_target_column="rpm_residual_oof",
            embed_mean=embed_mean,
            embed_std=embed_std,
            top_k=config.top_k,
            router_tag="train_case",
        )
        train_feature_rows.append(feature_row)
        neighbor_rows.extend(fold_neighbor_rows)

    holdout_feature_row, holdout_neighbor_rows = build_case_feature_row(
        target_row=holdout_row.iloc[0],
        candidate_df=train_case_df,
        base_pred_column="base_pred",
        residual_target_column=None,
        embed_mean=embed_mean,
        embed_std=embed_std,
        top_k=config.top_k,
        router_tag="holdout_case",
    )
    neighbor_rows.extend(holdout_neighbor_rows)

    train_feature_df = pd.DataFrame(train_feature_rows).sort_values("case_id").reset_index(drop=True)
    holdout_feature_df = pd.DataFrame([holdout_feature_row])

    delta_feature_columns = get_delta_only_feature_columns(train_feature_df)
    lowrank_train_df, lowrank_holdout_df = attach_delta_pca_features(
        train_feature_df=train_feature_df,
        holdout_feature_df=holdout_feature_df,
        n_components=config.delta_pca_components,
    )
    lowrank_feature_columns = get_lowrank_feature_columns(lowrank_train_df)

    case_rows = [
        build_case_row(
            domain_name,
            VARIANT_DELTA_ONLY,
            holdout,
            predict_with_bounded_head(
                train_feature_df=train_feature_df,
                holdout_feature_df=holdout_feature_df,
                feature_columns=delta_feature_columns,
                model_name=VARIANT_DELTA_ONLY,
                model_dir=model_dir,
                force_retrain=config.force_retrain,
                shrink=0.5,
            ),
        ),
        build_case_row(
            domain_name,
            VARIANT_LOWRANK,
            holdout,
            predict_with_bounded_head(
                train_feature_df=lowrank_train_df,
                holdout_feature_df=lowrank_holdout_df,
                feature_columns=lowrank_feature_columns,
                model_name=VARIANT_LOWRANK,
                model_dir=model_dir,
                force_retrain=config.force_retrain,
                shrink=0.5,
            ),
        ),
    ]

    all_feature_rows = [*train_feature_rows, holdout_feature_row]
    for row in all_feature_rows:
        row["holdout_case_id"] = holdout.case_id
        row["holdout_domain"] = domain_name
        row["row_role"] = "holdout" if int(row["case_id"]) == holdout.case_id else "train"

    delta_pca_rows = []
    for frame, role in ((lowrank_train_df, "train"), (lowrank_holdout_df, "holdout")):
        for _, row in frame.iterrows():
            out = row.to_dict()
            out["holdout_case_id"] = holdout.case_id
            out["holdout_domain"] = domain_name
            out["row_role"] = role
            delta_pca_rows.append(out)

    return case_rows, neighbor_rows, all_feature_rows, delta_pca_rows


def load_window_embeddings(
    *,
    try053,
    train_records,
    holdout,
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
    eval_dataset = try053.build_raw_window_dataset(
        [holdout],
        {holdout.case_id: cleaned_signal_frames[holdout.case_id]},
        window_config,
    )
    X_train = train_dataset.windows
    X_eval = eval_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)

    read_base = read_checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}"
    write_base = write_checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}"
    model = try053.TinyTCNEncoderRegressor(in_channels=X_train.shape[1])

    existing_paths = {
        "ckpt": read_base.with_suffix(".pt"),
        "norm": read_checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}_norm.npz",
        "meta": read_checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}.json",
    }
    cache_paths = {
        "ckpt": write_base.with_suffix(".pt"),
        "norm": write_checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}_norm.npz",
        "meta": write_checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}.json",
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
        X_train_norm, _, mean, std = try053.normalize_windows_by_channel(X_train, X_eval)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model.to(torch.device("cpu"))
        try053.train_model(model, X_train_norm, y_train, try053.TrainConfig(), torch.device("cpu"))
        torch.save(model.state_dict(), cache_paths["ckpt"])
        np.savez(cache_paths["norm"], mean=mean, std=std)
        cache_paths["meta"].write_text(
            json.dumps(
                {
                    "holdout_case_id": holdout.case_id,
                    "window_label": window_label,
                    "seed": seed,
                    "source": "058_fallback_train",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = model.cpu()

    X_train_norm = ((X_train - mean) / std).astype(np.float32)
    X_eval_norm = ((X_eval - mean) / std).astype(np.float32)
    with torch.no_grad():
        train_tensor = torch.from_numpy(X_train_norm).float()
        eval_tensor = torch.from_numpy(X_eval_norm).float()
        train_embedding = model.encode(train_tensor).cpu().numpy()
        eval_embedding = model.encode(eval_tensor).cpu().numpy()

    return {
        "train_meta_df": train_dataset.meta_df.copy(),
        "eval_meta_df": eval_dataset.meta_df.copy(),
        "train_embedding": train_embedding,
        "eval_embedding": eval_embedding,
    }


def build_case_table(records) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "rpm": float(record.rpm),
            }
            for record in records
        ]
    ).sort_values("case_id").reset_index(drop=True)


def add_rpm_oof_predictions(train_case_df: pd.DataFrame, try053) -> pd.DataFrame:
    preds: list[float] = []
    for case_id in train_case_df["case_id"]:
        inner_df = train_case_df.loc[train_case_df["case_id"] != case_id].copy()
        rpm_value = float(train_case_df.loc[train_case_df["case_id"] == case_id, "rpm"].iloc[0])
        preds.append(try053.predict_rpm_knn4_from_case_df(inner_df, rpm_value))
    result = train_case_df.copy()
    result["rpm_pred_oof"] = np.asarray(preds, dtype=float)
    result["rpm_residual_oof"] = result["true_wind_speed"] - result["rpm_pred_oof"]
    return result


def attach_case_embeddings(train_case_df: pd.DataFrame, per_window: dict[str, dict[str, object]]) -> pd.DataFrame:
    result = train_case_df.copy()
    for window_label in WINDOW_LABELS:
        case_embedding_df = aggregate_case_embeddings(
            per_window[window_label]["train_meta_df"],
            per_window[window_label]["train_embedding"],
            f"embedding_{window_label}",
        )
        result = result.merge(case_embedding_df, on="case_id", how="left")
    result["embedding_concat"] = result.apply(
        lambda row: np.concatenate(
            [
                np.asarray(row["embedding_2s"], dtype=float),
                np.asarray(row["embedding_8s"], dtype=float),
            ]
        ).astype(float),
        axis=1,
    )
    return result


def build_holdout_case_row(holdout, base_pred: float, per_window: dict[str, dict[str, object]]) -> pd.DataFrame:
    row = {
        "case_id": holdout.case_id,
        "file_name": holdout.file_name,
        "true_wind_speed": float(holdout.wind_speed),
        "rpm": float(holdout.rpm),
        "base_pred": float(base_pred),
    }
    for window_label in WINDOW_LABELS:
        eval_meta_df = per_window[window_label]["eval_meta_df"]
        eval_embedding = per_window[window_label]["eval_embedding"]
        row[f"embedding_{window_label}"] = eval_embedding[eval_meta_df.index.to_numpy(dtype=int, copy=False)].mean(axis=0).astype(float)
    row["embedding_concat"] = np.concatenate(
        [
            np.asarray(row["embedding_2s"], dtype=float),
            np.asarray(row["embedding_8s"], dtype=float),
        ]
    ).astype(float)
    return pd.DataFrame([row])


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


def build_case_feature_row(
    *,
    target_row: pd.Series,
    candidate_df: pd.DataFrame,
    base_pred_column: str,
    residual_target_column: str | None,
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    top_k: int,
    router_tag: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    local_pool_df = select_local_embedding_pool(
        candidate_df=candidate_df,
        target_row=target_row,
        embed_mean=embed_mean,
        embed_std=embed_std,
        top_k=top_k,
    )
    feature_row = build_prototype_feature_row(
        target_row=target_row,
        local_pool_df=local_pool_df,
        base_pred_column=base_pred_column,
        residual_target_column=residual_target_column,
    )

    neighbor_rows: list[dict[str, object]] = []
    for rank, (_, row) in enumerate(local_pool_df.iterrows(), start=1):
        neighbor_rows.append(
            {
                "router_tag": router_tag,
                "holdout_case_id": int(target_row["case_id"]),
                "target_case_id": int(target_row["case_id"]),
                "neighbor_rank": rank,
                "neighbor_case_id": int(row["case_id"]),
                "embedding_distance": float(row["embedding_distance"]),
                "weight": float(row["prototype_weight"]),
            }
        )

    feature_row["reference_case_ids"] = ",".join(str(int(case_id)) for case_id in local_pool_df["case_id"].tolist())
    return feature_row, neighbor_rows


def select_local_embedding_pool(
    *,
    candidate_df: pd.DataFrame,
    target_row: pd.Series,
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    support_matrix = np.vstack(candidate_df["embedding_concat"].to_numpy())
    target_vec = np.asarray(target_row["embedding_concat"], dtype=float)[None, :]
    scaled_support = (support_matrix - embed_mean) / embed_std
    scaled_target = (target_vec - embed_mean) / embed_std
    distances = np.sqrt(np.sum(np.square(scaled_support - scaled_target), axis=1))
    k = min(top_k, len(distances))
    order = np.argsort(distances)[:k]
    weights = 1.0 / np.maximum(distances[order], EPS)
    normalized_weights = weights / weights.sum()
    result = candidate_df.iloc[order].copy().reset_index(drop=True)
    result["embedding_distance"] = distances[order].astype(float)
    result["prototype_weight"] = normalized_weights.astype(float)
    return result


def build_prototype_feature_row(
    *,
    target_row: pd.Series,
    local_pool_df: pd.DataFrame,
    base_pred_column: str,
    residual_target_column: str | None,
) -> dict[str, object]:
    target_vec = np.asarray(target_row["embedding_concat"], dtype=float)
    support_matrix = np.vstack(local_pool_df["embedding_concat"].to_numpy())
    weights = local_pool_df["prototype_weight"].to_numpy(dtype=float, copy=False)
    ref_vec = np.average(support_matrix, axis=0, weights=weights)
    delta_vec = target_vec - ref_vec
    abs_delta_vec = np.abs(delta_vec)
    distances = local_pool_df["embedding_distance"].to_numpy(dtype=float, copy=False)

    feature_row: dict[str, object] = {
        "case_id": int(target_row["case_id"]),
        "file_name": str(target_row["file_name"]),
        "true_wind_speed": float(target_row["true_wind_speed"]),
        "rpm": float(target_row["rpm"]),
        "base_pred": float(target_row[base_pred_column]),
        "reference_pool_size": int(len(local_pool_df)),
        "top1_embed_distance": float(distances[0]),
        "topk_embed_mean_distance": float(np.average(distances, weights=weights)),
        "topk_embed_std_distance": float(
            np.sqrt(np.average(np.square(distances - np.average(distances, weights=weights)), weights=weights))
        ),
    }
    if residual_target_column is not None:
        feature_row["residual_target"] = float(target_row[residual_target_column])

    for index, value in enumerate(delta_vec, start=1):
        feature_row[f"delta_embed_{index}"] = float(value)
    for index, value in enumerate(abs_delta_vec, start=1):
        feature_row[f"abs_delta_embed_{index}"] = float(value)
    return feature_row


def attach_delta_pca_features(
    *,
    train_feature_df: pd.DataFrame,
    holdout_feature_df: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_columns = [column for column in train_feature_df.columns if column.startswith("delta_embed_")]
    train_delta = train_feature_df[delta_columns].to_numpy(dtype=float)
    holdout_delta = holdout_feature_df[delta_columns].to_numpy(dtype=float)
    effective_components = max(1, min(n_components, train_delta.shape[0] - 1, train_delta.shape[1]))
    pca = PCA(n_components=effective_components, random_state=42)
    train_proj = pca.fit_transform(train_delta)
    holdout_proj = pca.transform(holdout_delta)

    train_result = train_feature_df.copy()
    holdout_result = holdout_feature_df.copy()
    for index in range(effective_components):
        column = f"delta_pca_{index + 1}"
        train_result[column] = train_proj[:, index]
        holdout_result[column] = holdout_proj[:, index]
    return train_result, holdout_result


def get_delta_only_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    delta_columns = [column for column in feature_df.columns if column.startswith("delta_embed_")]
    abs_columns = [column for column in feature_df.columns if column.startswith("abs_delta_embed_")]
    dist_columns = [
        "base_pred",
        "reference_pool_size",
        "top1_embed_distance",
        "topk_embed_mean_distance",
        "topk_embed_std_distance",
    ]
    return [*delta_columns, *abs_columns, *dist_columns]


def get_lowrank_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [column for column in feature_df.columns if column.startswith("delta_pca_")]


def predict_with_bounded_head(
    *,
    train_feature_df: pd.DataFrame,
    holdout_feature_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    model_dir: Path,
    force_retrain: bool,
    shrink: float,
) -> float:
    case_id = int(holdout_feature_df["case_id"].iloc[0])
    model_path = model_dir / f"{model_name}_fold_case_{case_id}.pkl"
    meta_path = model_dir / f"{model_name}_fold_case_{case_id}.json"

    X_train = train_feature_df[feature_columns].to_numpy(dtype=float)
    y_train = train_feature_df["residual_target"].to_numpy(dtype=float)
    X_test = holdout_feature_df[feature_columns].to_numpy(dtype=float)
    bound = max(float(np.quantile(np.abs(y_train), 0.9)), 0.15)
    y_scaled = np.clip(y_train / bound, -0.999, 0.999)
    z_train = np.arctanh(y_scaled)

    if model_path.exists() and meta_path.exists() and not force_retrain:
        with model_path.open("rb") as f:
            model = pickle.load(f)
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
            ]
        )
        model.fit(X_train, z_train)
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "holdout_case_id": case_id,
                    "feature_columns": feature_columns,
                    "bound": bound,
                    "shrink": shrink,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    pred_latent = float(model.predict(X_test)[0])
    pred_delta = float(bound * np.tanh(pred_latent))
    base_pred = float(holdout_feature_df["base_pred"].iloc[0])
    return float(base_pred + shrink * pred_delta)


def build_case_row(domain_name: str, variant_name: str, record, pred_wind_speed: float) -> dict[str, object]:
    signed_error = float(pred_wind_speed - float(record.wind_speed))
    return {
        "domain": domain_name,
        "variant_name": variant_name,
        "case_id": record.case_id,
        "file_name": record.file_name,
        "true_wind_speed": float(record.wind_speed),
        "rpm": float(record.rpm),
        "pred_wind_speed": float(pred_wind_speed),
        "signed_error": signed_error,
        "abs_error": abs(signed_error),
    }


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, subset in (
        ("final_focus", case_level_df.loc[case_level_df["domain"] == "final_focus"].copy()),
        ("added_focus", case_level_df.loc[case_level_df["domain"] == "added_focus"].copy()),
        ("focus_all", case_level_df.copy()),
    ):
        for variant_name, block in subset.groupby("variant_name", sort=False):
            rows.append(
                {
                    "domain": domain_name,
                    "variant_name": variant_name,
                    "case_mae": float(block["abs_error"].mean()),
                    "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                    "mean_signed_error": float(block["signed_error"].mean()),
                    "case_count": int(len(block)),
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, summary_df: pd.DataFrame, case_level_df: pd.DataFrame) -> None:
    lines = [
        "# prototype head ablation quickcheck",
        "",
        f"- holdout 工况：`{', '.join(str(case_id) for case_id in HOLDOUT_CASE_IDS)}`",
        "",
        "## 三桶汇总",
        "",
    ]
    for domain_name in ("final_focus", "added_focus", "focus_all"):
        lines.append(f"### {domain_name}")
        lines.append("")
        subset = summary_df.loc[summary_df["domain"] == domain_name]
        for _, row in subset.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    lines.append("## 每工况重点对照")
    lines.append("")
    focus_variants = [
        "rpm_knn4",
        "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
        "rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5",
        "rpm_knn4__plus__embedding_topk_prototype_ridge_w0.5",
        VARIANT_DELTA_ONLY,
        VARIANT_LOWRANK,
    ]
    for case_id in HOLDOUT_CASE_IDS:
        lines.append(f"### 工况{case_id}")
        lines.append("")
        block = case_level_df.loc[
            (case_level_df["case_id"] == case_id) & (case_level_df["variant_name"].isin(focus_variants))
        ].sort_values(["abs_error", "variant_name"])
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: true=`{row['true_wind_speed']:.4f}`, "
                f"pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
