from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "074_external_059_delta_gate_replay"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
EMBEDDING_TABLE_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_case_table.csv"
TRY047_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "047_soft_gate_quickcheck" / "run_soft_gate_quickcheck.py"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DATA_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RPM_K = 4
TOP_K = 4
SHRINK = 0.5
ENABLE_THRESHOLD = 0.65
WEIGHT_BUCKETS = np.array([0.0, 0.3, 0.5, 1.0], dtype=float)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 external-first 口径下 replay 059 delta-only trigger/gate。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_df = pd.read_csv(EMBEDDING_TABLE_PATH, encoding="utf-8-sig")
    external_df = (
        embedding_df.loc[embedding_df["raw_source_domain"].isin(["added", "added2"]) & embedding_df["is_labeled"]]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    embedding_columns = [column for column in external_df.columns if column.startswith("embedding_")]

    try047 = load_module("try047_external_074", TRY047_SCRIPT_PATH)
    gate_feature_df = build_external_gate_feature_table(try047)
    gate_lookup = gate_feature_df.set_index("case_id")

    loocv_case_df, loocv_gate_train_df, loocv_proto_df, loocv_neighbor_df = run_external_loocv(
        external_df, embedding_columns, gate_lookup
    )
    transfer_case_df, transfer_gate_train_df, transfer_proto_df, transfer_neighbor_df = run_added_to_added2_transfer(
        external_df, embedding_columns, gate_lookup
    )

    all_case_df = pd.concat([loocv_case_df, transfer_case_df], ignore_index=True)
    all_gate_train_df = pd.concat([loocv_gate_train_df, transfer_gate_train_df], ignore_index=True)
    all_proto_df = pd.concat([loocv_proto_df, transfer_proto_df], ignore_index=True)
    all_neighbor_df = pd.concat([loocv_neighbor_df, transfer_neighbor_df], ignore_index=True)
    summary_by_protocol_df = build_summary_by_protocol(all_case_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(all_case_df)

    all_case_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(
        output_dir / "summary_by_protocol_and_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_feature_df.to_csv(output_dir / "gate_feature_table.csv", index=False, encoding="utf-8-sig")
    all_gate_train_df.to_csv(output_dir / "gate_training_table.csv", index=False, encoding="utf-8-sig")
    all_proto_df.to_csv(output_dir / "prototype_feature_table.csv", index=False, encoding="utf-8-sig")
    all_neighbor_df.to_csv(output_dir / "neighbor_table.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_by_protocol_df, summary_by_protocol_and_domain_df)

    best_row = summary_by_protocol_df.iloc[0]
    print("074 external 059 delta gate replay 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best protocol={best_row['protocol']} | "
        f"variant={best_row['variant_name']} | "
        f"case_mae={best_row['case_mae']:.4f}"
    )


def run_external_loocv(external_df: pd.DataFrame, embedding_columns: list[str], gate_lookup: pd.DataFrame):
    case_rows: list[dict[str, object]] = []
    gate_train_rows: list[dict[str, object]] = []
    proto_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for holdout_case_id in external_df["case_id"].tolist():
        train_df = external_df.loc[external_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_df = external_df.loc[external_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        fold_case_rows, fold_gate_train_rows, fold_proto_rows, fold_neighbor_rows = run_one_fold(
            protocol="external_loocv",
            train_df=train_df,
            test_df=test_df,
            embedding_columns=embedding_columns,
            gate_lookup=gate_lookup,
        )
        case_rows.extend(fold_case_rows)
        gate_train_rows.extend(fold_gate_train_rows)
        proto_rows.extend(fold_proto_rows)
        neighbor_rows.extend(fold_neighbor_rows)
    return pd.DataFrame(case_rows), pd.DataFrame(gate_train_rows), pd.DataFrame(proto_rows), pd.DataFrame(neighbor_rows)


def run_added_to_added2_transfer(external_df: pd.DataFrame, embedding_columns: list[str], gate_lookup: pd.DataFrame):
    train_df = external_df.loc[external_df["raw_source_domain"] == "added"].copy().reset_index(drop=True)
    case_rows: list[dict[str, object]] = []
    gate_train_rows: list[dict[str, object]] = []
    proto_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for holdout_case_id in external_df.loc[external_df["raw_source_domain"] == "added2", "case_id"].tolist():
        test_df = external_df.loc[external_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        fold_case_rows, fold_gate_train_rows, fold_proto_rows, fold_neighbor_rows = run_one_fold(
            protocol="added_to_added2",
            train_df=train_df,
            test_df=test_df,
            embedding_columns=embedding_columns,
            gate_lookup=gate_lookup,
        )
        case_rows.extend(fold_case_rows)
        gate_train_rows.extend(fold_gate_train_rows)
        proto_rows.extend(fold_proto_rows)
        neighbor_rows.extend(fold_neighbor_rows)
    return pd.DataFrame(case_rows), pd.DataFrame(gate_train_rows), pd.DataFrame(proto_rows), pd.DataFrame(neighbor_rows)


def run_one_fold(*, protocol: str, train_df: pd.DataFrame, test_df: pd.DataFrame, embedding_columns: list[str], gate_lookup: pd.DataFrame):
    train_base_preds = compute_internal_rpm_oof_predictions(train_df)
    train_feature_df, train_neighbor_rows = build_feature_table(
        feature_df=train_df,
        base_pred_values=train_base_preds,
        residual_targets=train_df["wind_speed"].to_numpy(dtype=float) - train_base_preds,
        embedding_columns=embedding_columns,
        protocol=protocol,
        row_role="train",
    )
    test_base_pred = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))
    test_feature_df, test_neighbor_rows = build_feature_table(
        feature_df=test_df,
        base_pred_values=test_base_pred,
        residual_targets=None,
        embedding_columns=embedding_columns,
        protocol=protocol,
        row_role="holdout",
        candidate_df=train_df,
    )

    delta_feature_columns = get_delta_only_feature_columns(train_feature_df)
    candidate_pred = predict_with_bounded_head(
        train_feature_df=train_feature_df,
        holdout_feature_df=test_feature_df,
        feature_columns=delta_feature_columns,
    )

    gate_train_rows: list[dict[str, object]] = []
    for row_idx in range(len(train_feature_df)):
        inner_train_feature_df = train_feature_df.drop(index=train_feature_df.index[row_idx]).reset_index(drop=True)
        inner_holdout_feature_df = train_feature_df.iloc[[row_idx]].reset_index(drop=True)
        inner_candidate_pred = predict_with_bounded_head(
            train_feature_df=inner_train_feature_df,
            holdout_feature_df=inner_holdout_feature_df,
            feature_columns=delta_feature_columns,
        )
        gate_train_rows.append(
            build_gate_row(
                protocol=protocol,
                feature_row=inner_holdout_feature_df.iloc[0],
                candidate_pred=inner_candidate_pred,
                gate_lookup=gate_lookup,
                true_wind_speed=float(inner_holdout_feature_df["true_wind_speed"].iloc[0]),
                row_role="train",
                holdout_case_id=int(test_df["case_id"].iloc[0]),
            )
        )

    train_gate_df = pd.DataFrame(gate_train_rows)
    holdout_gate_df = pd.DataFrame(
        [
            build_gate_row(
                protocol=protocol,
                feature_row=test_feature_df.iloc[0],
                candidate_pred=candidate_pred,
                gate_lookup=gate_lookup,
                true_wind_speed=float(test_df["wind_speed"].iloc[0]),
                row_role="holdout",
                holdout_case_id=int(test_df["case_id"].iloc[0]),
            )
        ]
    )
    gate_feature_columns = get_gate_feature_columns()

    base_pred = float(test_base_pred[0])
    case_rows = [
        build_case_row(protocol, "rpm_knn4", test_df.iloc[0], base_pred),
        build_case_row(protocol, "059_delta_only_prototype_ridge_w0.5", test_df.iloc[0], candidate_pred),
        build_case_row(
            protocol,
            "059_delta_only_soft_gate_hgb",
            test_df.iloc[0],
            predict_soft_gate_hgb(train_gate_df, holdout_gate_df, gate_feature_columns),
        ),
        build_case_row(
            protocol,
            f"059_delta_only_binary_hgb_t{ENABLE_THRESHOLD:.2f}",
            test_df.iloc[0],
            predict_binary_gate(train_gate_df, holdout_gate_df, gate_feature_columns, threshold=ENABLE_THRESHOLD),
        ),
        build_case_row(
            protocol,
            "059_delta_only_bucket_hgb",
            test_df.iloc[0],
            predict_bucket_gate(train_gate_df, holdout_gate_df, gate_feature_columns),
        ),
        build_case_row(
            protocol,
            f"059_delta_only_two_stage_hgb_t{ENABLE_THRESHOLD:.2f}",
            test_df.iloc[0],
            predict_two_stage_gate(train_gate_df, holdout_gate_df, gate_feature_columns, threshold=ENABLE_THRESHOLD),
        ),
        build_case_row(
            protocol,
            "059_delta_only_trigger_rule_cv",
            test_df.iloc[0],
            predict_trigger_rule(train_gate_df, holdout_gate_df),
        ),
    ]

    prototype_records = []
    for _, row in pd.concat([train_feature_df, test_feature_df], ignore_index=True).iterrows():
        record = row.to_dict()
        record["holdout_case_id"] = int(test_df["case_id"].iloc[0])
        prototype_records.append(record)
    return case_rows, gate_train_rows, prototype_records, [*train_neighbor_rows, *test_neighbor_rows]


def build_feature_table(
    *,
    feature_df: pd.DataFrame,
    base_pred_values: np.ndarray,
    residual_targets: np.ndarray | None,
    embedding_columns: list[str],
    protocol: str,
    row_role: str,
    candidate_df: pd.DataFrame | None = None,
):
    if candidate_df is None:
        candidate_df = feature_df
    rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []

    candidate_matrix = candidate_df[embedding_columns].to_numpy(dtype=float, copy=False)
    embed_mean = candidate_matrix.mean(axis=0, keepdims=True)
    embed_std = candidate_matrix.std(axis=0, keepdims=True)
    embed_std = np.where(embed_std > 0, embed_std, 1.0)

    for row_idx, (_, row) in enumerate(feature_df.iterrows()):
        if candidate_df is feature_df:
            local_candidate_df = candidate_df.drop(index=feature_df.index[row_idx]).reset_index(drop=True)
        else:
            local_candidate_df = candidate_df.copy().reset_index(drop=True)
        local_pool_df = select_local_embedding_pool(
            candidate_df=local_candidate_df,
            target_row=row,
            embedding_columns=embedding_columns,
            embed_mean=embed_mean,
            embed_std=embed_std,
            top_k=TOP_K,
        )
        feature_row = build_prototype_feature_row(
            target_row=row,
            local_pool_df=local_pool_df,
            base_pred=float(base_pred_values[row_idx]),
            residual_target=None if residual_targets is None else float(residual_targets[row_idx]),
            protocol=protocol,
            row_role=row_role,
        )
        rows.append(feature_row)
        for rank, (_, neighbor_row) in enumerate(local_pool_df.iterrows(), start=1):
            neighbor_rows.append(
                {
                    "protocol": protocol,
                    "row_role": row_role,
                    "holdout_case_id": int(row["case_id"]),
                    "target_case_id": int(row["case_id"]),
                    "neighbor_rank": rank,
                    "neighbor_case_id": int(neighbor_row["case_id"]),
                    "neighbor_domain": str(neighbor_row["raw_source_domain"]),
                    "embedding_distance": float(neighbor_row["embedding_distance"]),
                    "prototype_weight": float(neighbor_row["prototype_weight"]),
                }
            )
    return pd.DataFrame(rows), neighbor_rows


def select_local_embedding_pool(
    *,
    candidate_df: pd.DataFrame,
    target_row: pd.Series,
    embedding_columns: list[str],
    embed_mean: np.ndarray,
    embed_std: np.ndarray,
    top_k: int,
):
    support_matrix = candidate_df[embedding_columns].to_numpy(dtype=float, copy=False)
    target_vec = target_row[embedding_columns].to_numpy(dtype=float, copy=False)[None, :]
    scaled_support = (support_matrix - embed_mean) / embed_std
    scaled_target = (target_vec - embed_mean) / embed_std
    distances = np.sqrt(np.sum(np.square(scaled_support - scaled_target), axis=1))
    order = np.argsort(distances)[: min(top_k, len(candidate_df))]
    weights = 1.0 / np.maximum(distances[order], EPS)
    weights = weights / weights.sum()
    result = candidate_df.iloc[order].copy().reset_index(drop=True)
    result["embedding_distance"] = distances[order].astype(float)
    result["prototype_weight"] = weights.astype(float)
    return result


def build_prototype_feature_row(
    *,
    target_row: pd.Series,
    local_pool_df: pd.DataFrame,
    base_pred: float,
    residual_target: float | None,
    protocol: str,
    row_role: str,
):
    embedding_columns = [column for column in target_row.index if column.startswith("embedding_")]
    target_vec = target_row[embedding_columns].to_numpy(dtype=float, copy=False)
    support_matrix = local_pool_df[embedding_columns].to_numpy(dtype=float, copy=False)
    weights = local_pool_df["prototype_weight"].to_numpy(dtype=float, copy=False)
    ref_vec = np.average(support_matrix, axis=0, weights=weights)
    delta_vec = target_vec - ref_vec
    abs_delta_vec = np.abs(delta_vec)
    distances = local_pool_df["embedding_distance"].to_numpy(dtype=float, copy=False)
    row = {
        "protocol": protocol,
        "row_role": row_role,
        "case_id": int(target_row["case_id"]),
        "file_name": str(target_row["file_name"]),
        "domain": str(target_row["raw_source_domain"]),
        "true_wind_speed": float(target_row["wind_speed"]),
        "rpm": float(target_row["rpm"]),
        "base_pred": float(base_pred),
        "reference_pool_size": int(len(local_pool_df)),
        "top1_embed_distance": float(distances[0]),
        "topk_embed_mean_distance": float(np.average(distances, weights=weights)),
        "topk_embed_std_distance": float(
            np.sqrt(np.average(np.square(distances - np.average(distances, weights=weights)), weights=weights))
        ),
        "reference_case_ids": ",".join(str(int(case_id)) for case_id in local_pool_df["case_id"].tolist()),
    }
    if residual_target is not None:
        row["residual_target"] = float(residual_target)
    for idx, value in enumerate(delta_vec, start=1):
        row[f"delta_embed_{idx}"] = float(value)
    for idx, value in enumerate(abs_delta_vec, start=1):
        row[f"abs_delta_embed_{idx}"] = float(value)
    return row


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[float] = []
    for rpm_value in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm_value)
        order = np.argsort(distances)[: min(RPM_K, len(train_df))]
        weights = 1.0 / np.maximum(distances[order], EPS)
        weights = weights / weights.sum()
        predictions.append(float(np.dot(weights, train_wind[order])))
    return np.asarray(predictions, dtype=float)


def compute_internal_rpm_oof_predictions(train_df: pd.DataFrame) -> np.ndarray:
    if len(train_df) <= 1:
        return np.full(len(train_df), float(train_df["wind_speed"].mean()), dtype=float)
    predictions: list[float] = []
    rpm_values = train_df["rpm"].to_numpy(dtype=float, copy=False)
    for row_idx in range(len(train_df)):
        inner_train_df = train_df.drop(index=train_df.index[row_idx]).reset_index(drop=True)
        pred = compute_rpm_knn_predictions(inner_train_df, np.asarray([rpm_values[row_idx]], dtype=float))[0]
        predictions.append(float(pred))
    return np.asarray(predictions, dtype=float)


def get_delta_only_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    delta_columns = [column for column in feature_df.columns if column.startswith("delta_embed_")]
    abs_columns = [column for column in feature_df.columns if column.startswith("abs_delta_embed_")]
    return [
        *delta_columns,
        *abs_columns,
        "base_pred",
        "reference_pool_size",
        "top1_embed_distance",
        "topk_embed_mean_distance",
        "topk_embed_std_distance",
    ]


def predict_with_bounded_head(
    *,
    train_feature_df: pd.DataFrame,
    holdout_feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    X_train = train_feature_df[feature_columns].to_numpy(dtype=float)
    y_train = train_feature_df["residual_target"].to_numpy(dtype=float)
    X_test = holdout_feature_df[feature_columns].to_numpy(dtype=float)
    bound = max(float(np.quantile(np.abs(y_train), 0.9)), 0.15)
    y_scaled = np.clip(y_train / bound, -0.999, 0.999)
    z_train = np.arctanh(y_scaled)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])
    model.fit(X_train, z_train)
    pred_latent = float(model.predict(X_test)[0])
    pred_delta = float(bound * np.tanh(pred_latent))
    return float(holdout_feature_df["base_pred"].iloc[0] + SHRINK * pred_delta)


def build_external_gate_feature_table(try047) -> pd.DataFrame:
    external_records = [
        *load_manifest_records(ADDED_MANIFEST_PATH, ADDED_DATA_DIR),
        *load_manifest_records(ADDED2_MANIFEST_PATH, ADDED2_DATA_DIR),
    ]
    common_signal_columns = try047.get_common_signal_columns(external_records)
    gate_df = try047.build_gate_feature_table(external_records, common_signal_columns)
    gate_df["domain"] = gate_df["case_id"].map(lambda case_id: "added2" if int(case_id) >= 25 else "added")
    return gate_df.sort_values("case_id").reset_index(drop=True)


def load_manifest_records(manifest_path: Path, data_dir: Path):
    from src.current.data_loading import DatasetRecord

    manifest_df = pd.read_csv(manifest_path)
    records = []
    for _, row in manifest_df.iterrows():
        wind_speed = float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None
        rpm = float(row["rpm"]) if not pd.isna(row["rpm"]) else None
        records.append(
            DatasetRecord(
                case_id=int(row["case_id"]),
                display_name=str(row["display_name"]),
                file_name=f"工况{int(row['case_id'])}.csv",
                file_path=data_dir / f"工况{int(row['case_id'])}.csv",
                wind_speed=wind_speed,
                rpm=rpm,
                is_labeled=wind_speed is not None and rpm is not None,
                original_file_name=str(row.get("original_file_name", "")),
                label_source=str(row.get("label_source", "")),
                notes=str(row.get("notes", "")),
            )
        )
    return [record for record in records if record.is_labeled]


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_gate_row(
    *,
    protocol: str,
    feature_row: pd.Series,
    candidate_pred: float,
    gate_lookup: pd.DataFrame,
    true_wind_speed: float,
    row_role: str,
    holdout_case_id: int,
):
    case_id = int(feature_row["case_id"])
    gate_series = gate_lookup.loc[case_id]
    delta_values = np.asarray(
        [float(feature_row[column]) for column in feature_row.index if column.startswith("delta_embed_") and not column.startswith("abs_")],
        dtype=float,
    )
    abs_delta_values = np.abs(delta_values)
    base_pred = float(feature_row["base_pred"])
    pred_gap = float(candidate_pred - base_pred)
    optimal_gate = compute_optimal_gate_target(
        true_wind_speed=true_wind_speed,
        pred_base=base_pred,
        pred_enhanced=candidate_pred,
    )
    bucket_label = nearest_bucket(optimal_gate)
    return {
        "protocol": protocol,
        "holdout_case_id": int(holdout_case_id),
        "row_role": row_role,
        "domain": str(feature_row["domain"]),
        "case_id": case_id,
        "file_name": str(feature_row["file_name"]),
        "true_wind_speed": float(true_wind_speed),
        "true_rpm": float(feature_row["rpm"]),
        "base_pred": base_pred,
        "candidate_pred": float(candidate_pred),
        "pred_gap": pred_gap,
        "abs_pred_gap": abs(pred_gap),
        "top1_embed_distance": float(feature_row["top1_embed_distance"]),
        "topk_embed_mean_distance": float(feature_row["topk_embed_mean_distance"]),
        "topk_embed_std_distance": float(feature_row["topk_embed_std_distance"]),
        "delta_l2": float(np.sqrt(np.sum(np.square(delta_values)))),
        "delta_l1_mean": float(abs_delta_values.mean()),
        "delta_signed_mean": float(delta_values.mean()),
        "delta_max_abs": float(abs_delta_values.max()),
        "missing_ratio_in_common_cols": float(gate_series["missing_ratio_in_common_cols"]),
        "edge_removed_ratio": float(gate_series["edge_removed_ratio"]),
        "strain_low_over_mid": float(gate_series["strain_low_over_mid"]),
        "strain_mid_ratio_median": float(gate_series["strain_mid_ratio_median"]),
        "strain_rms_median": float(gate_series["strain_rms_median"]),
        "acc_energy_median": float(gate_series["acc_energy_median"]),
        "acc_peak_freq_median": float(gate_series["acc_peak_freq_median"]),
        "strain_acc_rms_ratio": float(gate_series["strain_acc_rms_ratio"]),
        "optimal_gate_target": float(optimal_gate),
        "bucket_label": float(bucket_label),
        "bucket_class": int(bucket_to_class(bucket_label)),
        "enable_enhanced": int(bucket_label > 0.0),
    }


def compute_optimal_gate_target(*, true_wind_speed: float, pred_base: float, pred_enhanced: float) -> float:
    denom = float(pred_enhanced - pred_base)
    if abs(denom) <= EPS:
        return 0.0
    return float(np.clip((float(true_wind_speed) - float(pred_base)) / denom, 0.0, 1.0))


def nearest_bucket(value: float) -> float:
    return float(WEIGHT_BUCKETS[np.argmin(np.abs(WEIGHT_BUCKETS - value))])


def bucket_to_class(bucket_value: float) -> int:
    return {0.0: 0, 0.3: 1, 0.5: 2, 1.0: 3}[float(bucket_value)]


def get_gate_feature_columns() -> list[str]:
    return [
        "true_rpm",
        "base_pred",
        "candidate_pred",
        "pred_gap",
        "abs_pred_gap",
        "top1_embed_distance",
        "topk_embed_mean_distance",
        "topk_embed_std_distance",
        "delta_l2",
        "delta_l1_mean",
        "delta_signed_mean",
        "delta_max_abs",
        "missing_ratio_in_common_cols",
        "edge_removed_ratio",
        "strain_low_over_mid",
        "strain_mid_ratio_median",
        "strain_rms_median",
        "acc_energy_median",
        "acc_peak_freq_median",
        "strain_acc_rms_ratio",
    ]


def predict_soft_gate_hgb(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> float:
    estimator = HistGradientBoostingRegressor(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=2,
        l2_regularization=0.1,
        random_state=42,
    )
    estimator.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["optimal_gate_target"].to_numpy(dtype=float))
    gate = float(np.clip(estimator.predict(test_df[feature_columns].to_numpy(dtype=float))[0], 0.0, 1.0))
    return blend_prediction(test_df, gate)


def predict_binary_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], threshold: float) -> float:
    unique_classes = np.unique(train_df["enable_enhanced"].to_numpy(dtype=int))
    if len(unique_classes) == 1:
        gate = float(unique_classes[0])
        return blend_prediction(test_df, gate)
    estimator = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=2,
        l2_regularization=0.1,
        random_state=42,
    )
    estimator.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["enable_enhanced"].to_numpy(dtype=int))
    prob = float(estimator.predict_proba(test_df[feature_columns].to_numpy(dtype=float))[0, 1])
    gate = 1.0 if prob >= threshold else 0.0
    return blend_prediction(test_df, gate)


def predict_bucket_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> float:
    unique_classes = np.unique(train_df["bucket_class"].to_numpy(dtype=int))
    if len(unique_classes) == 1:
        gate = {0: 0.0, 1: 0.3, 2: 0.5, 3: 1.0}[int(unique_classes[0])]
        return blend_prediction(test_df, gate)
    estimator = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(C=0.5, max_iter=1000))])
    estimator.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["bucket_class"].to_numpy(dtype=int))
    gate = {0: 0.0, 1: 0.3, 2: 0.5, 3: 1.0}[int(estimator.predict(test_df[feature_columns].to_numpy(dtype=float))[0])]
    return blend_prediction(test_df, gate)


def predict_two_stage_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], threshold: float) -> float:
    unique_enable = np.unique(train_df["enable_enhanced"].to_numpy(dtype=int))
    if len(unique_enable) == 1:
        if int(unique_enable[0]) == 0:
            return blend_prediction(test_df, 0.0)
        positive_train_df = train_df.copy()
    else:
        positive_train_df = None
    stage1 = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=2,
        l2_regularization=0.1,
        random_state=42,
    )
    if positive_train_df is None:
        stage1.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["enable_enhanced"].to_numpy(dtype=int))
        enable_prob = float(stage1.predict_proba(test_df[feature_columns].to_numpy(dtype=float))[0, 1])
        if enable_prob < threshold:
            return blend_prediction(test_df, 0.0)
        positive_train_df = train_df.loc[train_df["enable_enhanced"] == 1].copy()
    if positive_train_df.empty:
        return blend_prediction(test_df, 0.0)
    positive_classes = np.unique(positive_train_df["bucket_class"].clip(lower=1).to_numpy(dtype=int))
    if len(positive_classes) == 1:
        gate = {1: 0.3, 2: 0.5, 3: 1.0}[int(positive_classes[0])]
        return blend_prediction(test_df, gate)
    stage2 = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(C=0.5, max_iter=1000))])
    stage2.fit(
        positive_train_df[feature_columns].to_numpy(dtype=float),
        positive_train_df["bucket_class"].clip(lower=1).to_numpy(dtype=int),
    )
    gate = {1: 0.3, 2: 0.5, 3: 1.0}[int(stage2.predict(test_df[feature_columns].to_numpy(dtype=float))[0])]
    return blend_prediction(test_df, gate)


def predict_trigger_rule(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    best_score = float("inf")
    best_params = None
    for pos_gap in (0.15, 0.20, 0.25):
        for pos_top1 in (1.5, 2.0, 2.5):
            for neg_gap in (0.15, 0.20, 0.25):
                for neg_std in (1.5, 2.0, 2.5):
                    for strain_limit in (0.2, 1.0, 5.0):
                        pred = train_df.apply(
                            lambda row: apply_trigger_rule(
                                row=row,
                                positive_gap=pos_gap,
                                positive_top1=pos_top1,
                                negative_gap=neg_gap,
                                negative_std=neg_std,
                                strain_limit=strain_limit,
                            ),
                            axis=1,
                        )
                        mae = float(np.mean(np.abs(pred.to_numpy(dtype=float) - train_df["true_wind_speed"].to_numpy(dtype=float))))
                        if mae < best_score - 1e-12:
                            best_score = mae
                            best_params = (pos_gap, pos_top1, neg_gap, neg_std, strain_limit)
    if best_params is None:
        return blend_prediction(test_df, 0.0)
    gate = trigger_gate_value(
        row=test_df.iloc[0],
        positive_gap=best_params[0],
        positive_top1=best_params[1],
        negative_gap=best_params[2],
        negative_std=best_params[3],
        strain_limit=best_params[4],
    )
    return blend_prediction(test_df, gate)


def apply_trigger_rule(
    *,
    row: pd.Series,
    positive_gap: float,
    positive_top1: float,
    negative_gap: float,
    negative_std: float,
    strain_limit: float,
) -> float:
    gate = trigger_gate_value(
        row=row,
        positive_gap=positive_gap,
        positive_top1=positive_top1,
        negative_gap=negative_gap,
        negative_std=negative_std,
        strain_limit=strain_limit,
    )
    return float((1.0 - gate) * row["base_pred"] + gate * row["candidate_pred"])


def trigger_gate_value(
    *,
    row: pd.Series,
    positive_gap: float,
    positive_top1: float,
    negative_gap: float,
    negative_std: float,
    strain_limit: float,
) -> float:
    pred_gap = float(row["pred_gap"])
    abs_gap = abs(pred_gap)
    strain = float(row["strain_low_over_mid"])
    if pred_gap > 0.0:
        return 1.0 if abs_gap >= positive_gap and float(row["top1_embed_distance"]) <= positive_top1 and strain <= strain_limit else 0.0
    if pred_gap < 0.0:
        return 1.0 if abs_gap >= negative_gap and float(row["topk_embed_std_distance"]) <= negative_std and strain <= strain_limit else 0.0
    return 0.0


def blend_prediction(test_df: pd.DataFrame, gate: float) -> float:
    return float((1.0 - gate) * float(test_df["base_pred"].iloc[0]) + gate * float(test_df["candidate_pred"].iloc[0]))


def build_case_row(protocol: str, variant_name: str, row: pd.Series, pred_value: float) -> dict[str, object]:
    signed_error = float(pred_value - float(row["wind_speed"]))
    return {
        "protocol": protocol,
        "variant_name": variant_name,
        "case_id": int(row["case_id"]),
        "file_name": str(row["file_name"]),
        "domain": str(row["raw_source_domain"]),
        "true_wind_speed": float(row["wind_speed"]),
        "rpm": float(row["rpm"]),
        "pred_wind_speed": float(pred_value),
        "signed_error": signed_error,
        "abs_error": abs(signed_error),
    }


def summarize_block(block: pd.DataFrame) -> dict[str, object]:
    true_values = block["true_wind_speed"].to_numpy(dtype=float)
    pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    return {
        "case_count": int(len(block)),
        "case_mae": float(mean_absolute_error(true_values, pred_values)),
        "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mean_signed_error": float(np.mean(signed_error)),
        "max_abs_error": float(np.max(np.abs(signed_error))),
    }


def build_summary_by_protocol(all_case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, variant_name), block in all_case_df.groupby(["protocol", "variant_name"], sort=False):
        row = {"protocol": protocol, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_by_protocol_and_domain(all_case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, domain, variant_name), block in all_case_df.groupby(
        ["protocol", "domain", "variant_name"],
        sort=False,
    ):
        row = {"protocol": protocol, "domain": domain, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["protocol", "domain", "case_mae", "case_rmse", "variant_name"])
        .reset_index(drop=True)
    )


def write_summary_markdown(
    output_path: Path,
    summary_by_protocol_df: pd.DataFrame,
    summary_by_protocol_and_domain_df: pd.DataFrame,
) -> None:
    lines = ["# external 059 delta gate replay", "", "## Summary By Protocol", ""]
    for protocol, block in summary_by_protocol_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
            )
        lines.append("")
    lines.append("## Summary By Protocol And Domain")
    lines.append("")
    for (protocol, domain), block in summary_by_protocol_and_domain_df.groupby(["protocol", "domain"], sort=False):
        lines.append(f"### {protocol} | {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
