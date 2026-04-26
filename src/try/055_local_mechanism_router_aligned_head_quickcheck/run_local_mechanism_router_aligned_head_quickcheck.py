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
TRY_NAME = "055_local_mechanism_router_aligned_head_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY047_GATE_PATH = REPO_ROOT / "outputs" / "try" / "047_soft_gate_quickcheck" / "gate_feature_table.csv"
TRY052_CASE_PATH = REPO_ROOT / "outputs" / "try" / "052_tcn_embedding_window_signal_quickcheck" / "case_level_predictions.csv"
TRY053_CASE_PATH = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "case_level_predictions.csv"
TRY053_CKPT_DIR = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "models" / "checkpoints"
TRY053_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "053_support_window_residual_quickcheck" / "run_support_window_residual_quickcheck.py"
HOLDOUT_CASE_IDS = [1, 3, 17, 18, 21, 22, 23, 24]
WINDOW_LABELS = ("2s", "8s")
BASELINE_VARIANTS = [
    "rpm_knn4",
    "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
    "rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5",
]
MECHANISM_FEATURE_COLUMNS = [
    "missing_ratio_in_common_cols",
    "edge_removed_ratio",
    "strain_low_ratio_median",
    "strain_mid_ratio_median",
    "strain_low_over_mid",
    "strain_rms_median",
    "acc_energy_median",
    "acc_peak_freq_median",
    "strain_acc_rms_ratio",
    "hour_sin",
    "hour_cos",
]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
EPS = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    local_case_k: int = 6
    align_k: int = 4
    pca_components: int = 6
    random_seed: int = 42
    force_retrain: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="local mechanism router aligned head quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--local-case-k", type=int, default=6)
    parser.add_argument("--align-k", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=6)
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
        local_case_k=args.local_case_k,
        align_k=args.align_k,
        pca_components=args.pca_components,
        random_seed=args.random_seed,
        force_retrain=args.force_retrain,
    )

    try053 = load_try053_module()

    baseline_case_df = load_baseline_cases()
    gate_df = pd.read_csv(TRY047_GATE_PATH, encoding="utf-8-sig")

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
    router_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for holdout in holdout_records:
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        fold_case_rows, fold_router_rows, fold_feature_rows = run_holdout_fold(
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            gate_df=gate_df,
            domain_name=domain_by_case_id[holdout.case_id],
            model_dir=model_dir,
            encoder_cache_dir=encoder_cache_dir,
            config=config,
        )
        case_rows.extend(fold_case_rows)
        router_rows.extend(fold_router_rows)
        feature_rows.extend(fold_feature_rows)

    new_case_df = pd.DataFrame(case_rows)
    router_df = pd.DataFrame(router_rows)
    feature_df = pd.DataFrame(feature_rows)
    merged_case_df = pd.concat([baseline_case_df, new_case_df], ignore_index=True)
    summary_df = build_summary_by_domain(merged_case_df)

    merged_case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    router_df.to_csv(output_dir / "router_case_neighbors.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_dir / "aligned_feature_table.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, merged_case_df)

    best_focus = summary_df.loc[summary_df["domain"] == "focus_all"].iloc[0]
    print("055 local mechanism router aligned head quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best focus_all: {best_focus['variant_name']} | case_mae={best_focus['case_mae']:.4f}")


def load_try053_module():
    spec = importlib.util.spec_from_file_location("try053_support_module", TRY053_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 053 脚本: {TRY053_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["try053_support_module"] = module
    spec.loader.exec_module(module)
    return module


def load_baseline_cases() -> pd.DataFrame:
    case_052 = pd.read_csv(TRY052_CASE_PATH, encoding="utf-8-sig")
    case_053 = pd.read_csv(TRY053_CASE_PATH, encoding="utf-8-sig")
    frames = [
        case_052.loc[case_052["variant_name"].isin(BASELINE_VARIANTS[:2])].copy(),
        case_053.loc[case_053["variant_name"] == BASELINE_VARIANTS[2]].copy(),
    ]
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def run_holdout_fold(
    *,
    try053,
    train_records,
    holdout,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    gate_df: pd.DataFrame,
    domain_name: str,
    model_dir: Path,
    encoder_cache_dir: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
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

    train_case_df = train_case_df.merge(
        gate_df[["case_id", *MECHANISM_FEATURE_COLUMNS]],
        on="case_id",
        how="left",
    )
    holdout_features = gate_df.loc[gate_df["case_id"] == holdout.case_id, ["case_id", *MECHANISM_FEATURE_COLUMNS]].copy()
    holdout_row = holdout_row.merge(holdout_features, on="case_id", how="left")

    concat_matrix = np.vstack(train_case_df["embedding_concat"].to_numpy())
    embed_mean = concat_matrix.mean(axis=0, keepdims=True)
    embed_std = concat_matrix.std(axis=0, keepdims=True)
    embed_std = np.where(embed_std > 0, embed_std, 1.0)
    pca_components = min(config.pca_components, concat_matrix.shape[0] - 1, concat_matrix.shape[1])
    pca = PCA(n_components=max(pca_components, 1), random_state=config.random_seed + holdout.case_id)
    pca.fit(concat_matrix)

    train_feature_rows: list[dict[str, object]] = []
    router_rows: list[dict[str, object]] = []
    for _, target_row in train_case_df.iterrows():
        candidate_df = train_case_df.loc[train_case_df["case_id"] != int(target_row["case_id"])].reset_index(drop=True)
        feature_row, fold_router_rows = build_case_feature_row(
            target_row=target_row,
            candidate_df=candidate_df,
            base_pred_column="rpm_pred_oof",
            residual_target_column="rpm_residual_oof",
            pca=pca,
            embed_mean=embed_mean,
            embed_std=embed_std,
            local_case_k=config.local_case_k,
            align_k=config.align_k,
            router_tag="train_case",
        )
        train_feature_rows.append(feature_row)
        router_rows.extend(fold_router_rows)

    holdout_feature_row, holdout_router_rows = build_case_feature_row(
        target_row=holdout_row.iloc[0],
        candidate_df=train_case_df,
        base_pred_column="base_pred",
        residual_target_column=None,
        pca=pca,
        embed_mean=embed_mean,
        embed_std=embed_std,
        local_case_k=config.local_case_k,
        align_k=config.align_k,
        router_tag="holdout_case",
    )
    router_rows.extend(holdout_router_rows)

    train_feature_df = pd.DataFrame(train_feature_rows).sort_values("case_id").reset_index(drop=True)
    holdout_feature_df = pd.DataFrame([holdout_feature_row])

    head_variants = {
        "rpm_knn4__plus__local_mechanism_aligned_tanh_ridge_pca6": 1.0,
        "rpm_knn4__plus__local_mechanism_aligned_tanh_ridge_pca6_w0.5": 0.5,
    }
    feature_columns = get_head_feature_columns(train_feature_df)

    case_rows = [
        build_case_row(
            domain_name,
            "rpm_knn4__plus__local_mechanism_residual_knn4_concat_2s_8s_w0.5",
            holdout,
            float(holdout_feature_row["local_residual_pred"]),
        ),
    ]
    for variant_name, shrink in head_variants.items():
        pred = predict_with_bounded_head(
            train_feature_df=train_feature_df,
            holdout_feature_df=holdout_feature_df,
            feature_columns=feature_columns,
            model_name=variant_name,
            model_dir=model_dir,
            force_retrain=config.force_retrain,
            shrink=shrink,
        )
        case_rows.append(build_case_row(domain_name, variant_name, holdout, pred))

    all_feature_rows = [*train_feature_rows, holdout_feature_row]
    for row in all_feature_rows:
        row["holdout_case_id"] = holdout.case_id
        row["holdout_domain"] = domain_name
        row["row_role"] = "holdout" if int(row["case_id"]) == holdout.case_id else "train"

    return case_rows, router_rows, all_feature_rows


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
                    "source": "055_fallback_train",
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
    pca: PCA,
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    local_case_k: int,
    align_k: int,
    router_tag: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    local_pool_df = select_local_mechanism_pool(candidate_df, target_row, local_case_k=local_case_k)
    local_residual_pred, residual_neighbor_rows = predict_local_mechanism_residual(
        target_row=target_row,
        local_pool_df=local_pool_df,
        embed_mean=embed_mean,
        embed_std=embed_std,
        base_pred_column=base_pred_column,
        align_k=align_k,
    )
    feature_row, aligned_neighbor_rows = build_aligned_feature_row(
        target_row=target_row,
        local_pool_df=local_pool_df,
        pca=pca,
        embed_mean=embed_mean,
        embed_std=embed_std,
        base_pred_column=base_pred_column,
        residual_target_column=residual_target_column,
        align_k=align_k,
    )

    router_rows = []
    for rank, (_, pool_row) in enumerate(local_pool_df.iterrows(), start=1):
        router_rows.append(
            {
                "router_tag": router_tag,
                "holdout_case_id": int(target_row["case_id"]),
                "target_case_id": int(target_row["case_id"]),
                "stage": "mechanism_pool",
                "neighbor_rank": rank,
                "neighbor_case_id": int(pool_row["case_id"]),
                "mechanism_distance": float(pool_row["mechanism_distance"]),
                "embedding_distance": np.nan,
                "weight": np.nan,
            }
        )
    router_rows.extend(
        [
            {
                "router_tag": router_tag,
                "holdout_case_id": int(target_row["case_id"]),
                "target_case_id": int(target_row["case_id"]),
                "stage": "aligned_neighbor",
                "neighbor_rank": int(row["neighbor_rank"]),
                "neighbor_case_id": int(row["neighbor_case_id"]),
                "mechanism_distance": float(row["mechanism_distance"]),
                "embedding_distance": float(row["embedding_distance"]),
                "weight": float(row["weight"]),
            }
            for row in aligned_neighbor_rows
        ]
    )
    router_rows.extend(
        [
            {
                "router_tag": router_tag,
                "holdout_case_id": int(target_row["case_id"]),
                "target_case_id": int(target_row["case_id"]),
                "stage": "local_residual_neighbor",
                "neighbor_rank": int(row["neighbor_rank"]),
                "neighbor_case_id": int(row["neighbor_case_id"]),
                "mechanism_distance": float(row["mechanism_distance"]),
                "embedding_distance": float(row["embedding_distance"]),
                "weight": float(row["weight"]),
            }
            for row in residual_neighbor_rows
        ]
    )
    feature_row["local_residual_pred"] = float(local_residual_pred)
    feature_row["local_pool_case_ids"] = ",".join(str(int(case_id)) for case_id in local_pool_df["case_id"].tolist())
    return feature_row, router_rows


def select_local_mechanism_pool(candidate_df: pd.DataFrame, target_row: pd.Series, *, local_case_k: int) -> pd.DataFrame:
    matrix = candidate_df[MECHANISM_FEATURE_COLUMNS].to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    target_vec = target_row[MECHANISM_FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
    distances = np.sqrt(np.sum(np.square((matrix - target_vec) / std), axis=1))
    result = candidate_df.copy()
    result["mechanism_distance"] = distances.astype(float)
    return result.nsmallest(min(local_case_k, len(result)), "mechanism_distance").reset_index(drop=True)


def predict_local_mechanism_residual(
    *,
    target_row: pd.Series,
    local_pool_df: pd.DataFrame,
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    base_pred_column: str,
    align_k: int,
) -> tuple[float, list[dict[str, object]]]:
    support_matrix = np.vstack(local_pool_df["embedding_concat"].to_numpy())
    target_vec = np.asarray(target_row["embedding_concat"], dtype=float)[None, :]
    scaled_support = (support_matrix - embed_mean) / embed_std
    scaled_target = (target_vec - embed_mean) / embed_std
    distances = np.sqrt(np.sum(np.square(scaled_support - scaled_target), axis=1))
    k = min(align_k, len(distances))
    order = np.argsort(distances)[:k]
    weights = 1.0 / np.maximum(distances[order], EPS)
    residual_target = local_pool_df["rpm_residual_oof"].to_numpy(dtype=float, copy=False)
    residual_pred = float(np.average(residual_target[order], weights=weights))
    clip_value = max(float(np.quantile(np.abs(local_pool_df["rpm_residual_oof"]), 0.9)), 0.15)
    residual_pred = float(np.clip(residual_pred, -clip_value, clip_value))
    base_pred = float(target_row[base_pred_column])
    pred = float(base_pred + 0.5 * residual_pred)

    rows: list[dict[str, object]] = []
    weight_sum = weights.sum()
    for rank, idx in enumerate(order, start=1):
        pool_row = local_pool_df.iloc[idx]
        rows.append(
            {
                "neighbor_rank": rank,
                "neighbor_case_id": int(pool_row["case_id"]),
                "mechanism_distance": float(pool_row["mechanism_distance"]),
                "embedding_distance": float(distances[idx]),
                "weight": float(weights[rank - 1] / weight_sum),
            }
        )
    return pred, rows


def build_aligned_feature_row(
    *,
    target_row: pd.Series,
    local_pool_df: pd.DataFrame,
    pca: PCA,
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    base_pred_column: str,
    residual_target_column: str | None,
    align_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    support_matrix = np.vstack(local_pool_df["embedding_concat"].to_numpy())
    target_vec = np.asarray(target_row["embedding_concat"], dtype=float)
    scaled_support = (support_matrix - embed_mean) / embed_std
    scaled_target = (target_vec[None, :] - embed_mean) / embed_std
    distances = np.sqrt(np.sum(np.square(scaled_support - scaled_target), axis=1))
    k = min(align_k, len(distances))
    order = np.argsort(distances)[:k]
    weights = 1.0 / np.maximum(distances[order], EPS)
    normalized_weights = weights / weights.sum()

    top_support = local_pool_df.iloc[order].reset_index(drop=True)
    support_mean_concat = np.average(
        np.vstack(top_support["embedding_concat"].to_numpy()),
        axis=0,
        weights=normalized_weights,
    )
    target_pca = pca.transform(target_vec.reshape(1, -1))[0]
    support_pca = pca.transform(support_mean_concat.reshape(1, -1))[0]
    delta_pca = target_pca - support_pca

    support_base_mean = float(np.average(top_support["rpm_pred_oof"].to_numpy(dtype=float), weights=normalized_weights))
    support_rpm_mean = float(np.average(top_support["rpm"].to_numpy(dtype=float), weights=normalized_weights))
    mechanism_weighted_mean = np.average(
        top_support[MECHANISM_FEATURE_COLUMNS].to_numpy(dtype=float),
        axis=0,
        weights=normalized_weights,
    )
    mechanism_delta = target_row[MECHANISM_FEATURE_COLUMNS].to_numpy(dtype=float) - mechanism_weighted_mean

    feature_row: dict[str, object] = {
        "case_id": int(target_row["case_id"]),
        "file_name": str(target_row["file_name"]),
        "true_wind_speed": float(target_row["true_wind_speed"]),
        "rpm": float(target_row["rpm"]),
        "base_pred": float(target_row[base_pred_column]),
        "top1_embed_distance": float(distances[order[0]]),
        "topk_embed_mean_distance": float(np.average(distances[order], weights=normalized_weights)),
        "topk_embed_std_distance": float(
            np.sqrt(
                np.average(
                    np.square(distances[order] - np.average(distances[order], weights=normalized_weights)),
                    weights=normalized_weights,
                )
            )
        ),
        "local_mechanism_mean_distance": float(np.mean(local_pool_df["mechanism_distance"])),
        "local_mechanism_min_distance": float(np.min(local_pool_df["mechanism_distance"])),
        "support_base_mean": support_base_mean,
        "support_base_gap": float(target_row[base_pred_column] - support_base_mean),
        "support_rpm_mean": support_rpm_mean,
        "support_rpm_gap": float(target_row["rpm"] - support_rpm_mean),
    }
    if residual_target_column is not None:
        feature_row["residual_target"] = float(target_row[residual_target_column])

    for index, value in enumerate(target_pca, start=1):
        feature_row[f"target_pca_{index}"] = float(value)
    for index, value in enumerate(support_pca, start=1):
        feature_row[f"support_pca_{index}"] = float(value)
    for index, value in enumerate(delta_pca, start=1):
        feature_row[f"delta_pca_{index}"] = float(value)
    for name, value in zip(MECHANISM_FEATURE_COLUMNS, target_row[MECHANISM_FEATURE_COLUMNS].to_numpy(dtype=float), strict=True):
        feature_row[f"target::{name}"] = float(value)
    for name, value in zip(MECHANISM_FEATURE_COLUMNS, mechanism_delta, strict=True):
        feature_row[f"delta::{name}"] = float(value)

    neighbor_rows: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        pool_row = local_pool_df.iloc[idx]
        neighbor_rows.append(
            {
                "neighbor_rank": rank,
                "neighbor_case_id": int(pool_row["case_id"]),
                "mechanism_distance": float(pool_row["mechanism_distance"]),
                "embedding_distance": float(distances[idx]),
                "weight": float(normalized_weights[rank - 1]),
            }
        )
    return feature_row, neighbor_rows


def get_head_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    ignore = {
        "case_id",
        "file_name",
        "true_wind_speed",
        "rpm",
        "base_pred",
        "residual_target",
        "local_residual_pred",
        "local_pool_case_ids",
        "holdout_case_id",
        "holdout_domain",
        "row_role",
    }
    return [column for column in feature_df.columns if column not in ignore]


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
        "# local mechanism router aligned head quickcheck",
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
        "rpm_knn4__plus__local_mechanism_residual_knn4_concat_2s_8s_w0.5",
        "rpm_knn4__plus__local_mechanism_aligned_tanh_ridge_pca6",
        "rpm_knn4__plus__local_mechanism_aligned_tanh_ridge_pca6_w0.5",
    ]
    for case_id in HOLDOUT_CASE_IDS:
        lines.append(f"### 工况{case_id}")
        lines.append("")
        block = case_level_df.loc[(case_level_df["case_id"] == case_id) & (case_level_df["variant_name"].isin(focus_variants))]
        block = block.sort_values(["abs_error", "variant_name"])
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: true=`{row['true_wind_speed']:.4f}`, "
                f"pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
