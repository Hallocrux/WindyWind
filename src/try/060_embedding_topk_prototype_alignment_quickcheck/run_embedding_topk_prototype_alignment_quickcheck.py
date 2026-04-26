from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "060_embedding_topk_prototype_alignment_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY058_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "058_prototype_head_ablation_quickcheck" / "run_prototype_head_ablation_quickcheck.py"
TRY058_CASE_PATH = REPO_ROOT / "outputs" / "try" / "058_prototype_head_ablation_quickcheck" / "case_level_predictions.csv"

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
EPS = 1e-6
ALIGN_TEMPERATURE = 2.0
PCA_COMPONENTS = 6
HEAD_VARIANTS = {
    "rpm_knn4__plus__embedding_prototype_alignment_ridge": 1.0,
    "rpm_knn4__plus__embedding_prototype_alignment_ridge_w0.5": 0.5,
}
BASELINE_VARIANTS = [
    "rpm_knn4",
    "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
    "rpm_knn4__plus__delta_only_prototype_ridge_w0.5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="embedding top-k prototype alignment quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
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

    try058 = load_module("try058_proto_align", TRY058_SCRIPT_PATH)
    try053 = try058.load_try053_module()

    baseline_case_df = load_baseline_cases()

    final_records = [record for record in try053.scan_dataset_records() if record.is_labeled]
    added_records = try053.load_added_records()
    all_records = sorted([*final_records, *added_records], key=lambda record: record.case_id)
    record_by_case_id = {record.case_id: record for record in all_records}
    holdout_records = [record_by_case_id[case_id] for case_id in try058.HOLDOUT_CASE_IDS]
    domain_by_case_id = {case_id: ("final_focus" if case_id < 21 else "added_focus") for case_id in try058.HOLDOUT_CASE_IDS}

    common_signal_columns = try053.get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: try053.load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for holdout in holdout_records:
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        fold_case_rows, fold_neighbor_rows, fold_feature_rows = run_holdout_fold(
            try058=try058,
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            domain_name=domain_by_case_id[holdout.case_id],
            model_dir=model_dir,
            encoder_cache_dir=encoder_cache_dir,
            top_k=args.top_k,
            random_seed=args.random_seed,
            force_retrain=args.force_retrain,
        )
        case_rows.extend(fold_case_rows)
        neighbor_rows.extend(fold_neighbor_rows)
        feature_rows.extend(fold_feature_rows)

    new_case_df = pd.DataFrame(case_rows)
    neighbor_df = pd.DataFrame(neighbor_rows)
    feature_df = pd.DataFrame(feature_rows)
    merged_case_df = pd.concat([baseline_case_df, new_case_df], ignore_index=True)
    summary_df = build_summary_by_domain(merged_case_df)

    merged_case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(output_dir / "prototype_neighbors.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_dir / "alignment_feature_table.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, merged_case_df)

    best_focus = summary_df.loc[summary_df["domain"] == "focus_all"].iloc[0]
    print("060 embedding top-k prototype alignment quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best focus_all: {best_focus['variant_name']} | case_mae={best_focus['case_mae']:.4f}")


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline_cases() -> pd.DataFrame:
    case_df = pd.read_csv(TRY058_CASE_PATH, encoding="utf-8-sig")
    return case_df.loc[case_df["variant_name"].isin(BASELINE_VARIANTS)].copy().reset_index(drop=True)


def run_holdout_fold(
    *,
    try058,
    try053,
    train_records,
    holdout,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    domain_name: str,
    model_dir: Path,
    encoder_cache_dir: Path,
    top_k: int,
    random_seed: int,
    force_retrain: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_case_df = try058.build_case_table(train_records)
    train_case_df = try058.add_rpm_oof_predictions(train_case_df, try053)
    holdout_base_pred = float(try053.predict_rpm_knn4_with_neighbors(train_records, float(holdout.rpm))[0])

    per_window: dict[str, dict[str, object]] = {}
    for order, window_label in enumerate(try058.WINDOW_LABELS, start=1):
        window_config = try053.WINDOW_CONFIGS[window_label]
        per_window[window_label] = try058.load_window_embeddings(
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            window_label=window_label,
            seed=random_seed + holdout.case_id * 100 + order,
            read_checkpoint_dir=try058.TRY053_CKPT_DIR,
            write_checkpoint_dir=encoder_cache_dir,
            force_retrain=force_retrain,
        )

    train_case_df = try058.attach_case_embeddings(train_case_df, per_window)
    holdout_row = try058.build_holdout_case_row(holdout, holdout_base_pred, per_window)

    concat_matrix = np.vstack(train_case_df["embedding_concat"].to_numpy())
    embed_mean = concat_matrix.mean(axis=0, keepdims=True)
    embed_std = concat_matrix.std(axis=0, keepdims=True)
    embed_std = np.where(embed_std > 0, embed_std, 1.0)
    pca = PCA(
        n_components=max(1, min(PCA_COMPONENTS, concat_matrix.shape[0] - 1, concat_matrix.shape[1])),
        random_state=random_seed + holdout.case_id,
    )
    pca.fit(concat_matrix)

    train_feature_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for _, target_row in train_case_df.iterrows():
        candidate_df = train_case_df.loc[train_case_df["case_id"] != int(target_row["case_id"])].reset_index(drop=True)
        feature_row, fold_neighbor_rows = build_alignment_feature_row(
            target_row=target_row,
            candidate_df=candidate_df,
            base_pred_column="rpm_pred_oof",
            residual_target_column="rpm_residual_oof",
            embed_mean=embed_mean,
            embed_std=embed_std,
            top_k=top_k,
            pca=pca,
            router_tag="train_case",
        )
        train_feature_rows.append(feature_row)
        neighbor_rows.extend(fold_neighbor_rows)

    holdout_feature_row, holdout_neighbor_rows = build_alignment_feature_row(
        target_row=holdout_row.iloc[0],
        candidate_df=train_case_df,
        base_pred_column="base_pred",
        residual_target_column=None,
        embed_mean=embed_mean,
        embed_std=embed_std,
        top_k=top_k,
        pca=pca,
        router_tag="holdout_case",
    )
    neighbor_rows.extend(holdout_neighbor_rows)

    train_feature_df = pd.DataFrame(train_feature_rows).sort_values("case_id").reset_index(drop=True)
    holdout_feature_df = pd.DataFrame([holdout_feature_row])
    feature_columns = get_head_feature_columns(train_feature_df)

    case_rows: list[dict[str, object]] = []
    for variant_name, shrink in HEAD_VARIANTS.items():
        pred = predict_with_bounded_head(
            train_feature_df=train_feature_df,
            holdout_feature_df=holdout_feature_df,
            feature_columns=feature_columns,
            model_name=variant_name,
            model_dir=model_dir,
            force_retrain=force_retrain,
            shrink=shrink,
        )
        case_rows.append(build_case_row(domain_name, variant_name, holdout, pred))

    all_feature_rows = [*train_feature_rows, holdout_feature_row]
    for row in all_feature_rows:
        row["holdout_case_id"] = holdout.case_id
        row["holdout_domain"] = domain_name
        row["row_role"] = "holdout" if int(row["case_id"]) == holdout.case_id else "train"

    return case_rows, neighbor_rows, all_feature_rows


def build_alignment_feature_row(
    *,
    target_row: pd.Series,
    candidate_df: pd.DataFrame,
    base_pred_column: str,
    residual_target_column: str | None,
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    top_k: int,
    pca: PCA,
    router_tag: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    local_pool_df = try_select_local_embedding_pool(
        candidate_df=candidate_df,
        target_row=target_row,
        embed_mean=embed_mean,
        embed_std=embed_std,
        top_k=top_k,
    )
    feature_row = build_prototype_alignment_features(
        target_row=target_row,
        local_pool_df=local_pool_df,
        base_pred_column=base_pred_column,
        residual_target_column=residual_target_column,
        pca=pca,
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


def try_select_local_embedding_pool(
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


def build_prototype_alignment_features(
    *,
    target_row: pd.Series,
    local_pool_df: pd.DataFrame,
    base_pred_column: str,
    residual_target_column: str | None,
    pca: PCA,
) -> dict[str, object]:
    target_vec = np.asarray(target_row["embedding_concat"], dtype=float)
    support_matrix = np.vstack(local_pool_df["embedding_concat"].to_numpy())
    weights = local_pool_df["prototype_weight"].to_numpy(dtype=float, copy=False)
    ref_vec = np.average(support_matrix, axis=0, weights=weights)
    centered_support = support_matrix - ref_vec[None, :]
    local_var = np.average(np.square(centered_support), axis=0, weights=weights)
    local_std = np.sqrt(np.maximum(local_var, EPS))
    scale_floor = max(float(np.quantile(local_std, 0.25)), 0.05)
    effective_scale = np.maximum(local_std, scale_floor)

    delta_vec = target_vec - ref_vec
    delta_z = delta_vec / effective_scale
    aligned_delta = np.tanh(delta_z / ALIGN_TEMPERATURE) * effective_scale
    aligned_vec = ref_vec + aligned_delta

    ref_pca = pca.transform(ref_vec.reshape(1, -1))[0]
    aligned_pca = pca.transform(aligned_vec.reshape(1, -1))[0]
    aligned_delta_pca = aligned_pca - ref_pca

    distances = local_pool_df["embedding_distance"].to_numpy(dtype=float, copy=False)
    cosine = float(np.dot(target_vec, ref_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(ref_vec) + EPS))
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
        "prototype_dispersion_mean": float(local_std.mean()),
        "prototype_dispersion_max": float(local_std.max()),
        "prototype_dispersion_min": float(local_std.min()),
        "delta_z_l2": float(np.sqrt(np.sum(np.square(delta_z)))),
        "delta_z_abs_mean": float(np.abs(delta_z).mean()),
        "delta_z_abs_max": float(np.abs(delta_z).max()),
        "raw_delta_l2": float(np.sqrt(np.sum(np.square(delta_vec)))),
        "raw_delta_abs_mean": float(np.abs(delta_vec).mean()),
        "prototype_cosine_similarity": cosine,
    }
    if residual_target_column is not None:
        feature_row["residual_target"] = float(target_row[residual_target_column])

    for index, value in enumerate(ref_pca, start=1):
        feature_row[f"prototype_pca_{index}"] = float(value)
    for index, value in enumerate(aligned_delta_pca, start=1):
        feature_row[f"aligned_delta_pca_{index}"] = float(value)
    return feature_row


def get_head_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    ignore = {
        "case_id",
        "file_name",
        "true_wind_speed",
        "rpm",
        "residual_target",
        "reference_case_ids",
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
        "# embedding top-k prototype alignment quickcheck",
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

    focus_variants = [*BASELINE_VARIANTS, *HEAD_VARIANTS.keys()]
    lines.append("## 每工况重点对照")
    lines.append("")
    for case_id in sorted(case_level_df["case_id"].unique()):
        lines.append(f"### 工况{int(case_id)}")
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
