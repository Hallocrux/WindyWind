from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "059_delta_only_gate_bucket_trigger_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY047_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "047_soft_gate_quickcheck" / "run_soft_gate_quickcheck.py"
TRY058_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "058_prototype_head_ablation_quickcheck" / "run_prototype_head_ablation_quickcheck.py"
TRY058_CASE_PATH = REPO_ROOT / "outputs" / "try" / "058_prototype_head_ablation_quickcheck" / "case_level_predictions.csv"

ENABLE_THRESHOLD = 0.65
WEIGHT_BUCKETS = np.array([0.0, 0.3, 0.5, 1.0], dtype=float)
POSITIVE_WEIGHT_BUCKETS = np.array([0.3, 0.5, 1.0], dtype=float)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="delta-only prototype 上的 gate / bucket / trigger quickcheck。")
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

    try047 = load_module("try047_delta_gate", TRY047_SCRIPT_PATH)
    try058 = load_module("try058_delta_gate", TRY058_SCRIPT_PATH)
    try053 = try058.load_try053_module()

    base_case_df = load_reference_case_rows()
    gate_lookup = build_gate_lookup(try047)

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
    gate_train_rows: list[dict[str, object]] = []
    gate_feature_rows: list[dict[str, object]] = []
    for holdout in holdout_records:
        outer_dir = model_dir / f"outer_case_{holdout.case_id}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        fold_case_rows, fold_gate_train_rows, fold_gate_feature_rows = run_outer_fold(
            try047=try047,
            try058=try058,
            try053=try053,
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            gate_lookup=gate_lookup,
            domain_name=domain_by_case_id[holdout.case_id],
            outer_model_dir=outer_dir,
            encoder_cache_dir=encoder_cache_dir,
            top_k=args.top_k,
            random_seed=args.random_seed,
            force_retrain=args.force_retrain,
        )
        case_rows.extend(fold_case_rows)
        gate_train_rows.extend(fold_gate_train_rows)
        gate_feature_rows.extend(fold_gate_feature_rows)

    new_case_df = pd.DataFrame(case_rows)
    merged_case_df = pd.concat([base_case_df, new_case_df], ignore_index=True)
    summary_df = build_summary_by_domain(merged_case_df)
    gate_train_df = pd.DataFrame(gate_train_rows)
    gate_feature_df = pd.DataFrame(gate_feature_rows)

    merged_case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    gate_train_df.to_csv(output_dir / "gate_training_table.csv", index=False, encoding="utf-8-sig")
    gate_feature_df.to_csv(output_dir / "gate_feature_table.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, merged_case_df)

    best_focus = summary_df.loc[summary_df["domain"] == "focus_all"].iloc[0]
    print("059 delta-only gate / bucket / trigger quickcheck 已完成。")
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


def load_reference_case_rows() -> pd.DataFrame:
    case_df = pd.read_csv(TRY058_CASE_PATH, encoding="utf-8-sig")
    keep_variants = [
        "rpm_knn4",
        "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
        "rpm_knn4__plus__delta_only_prototype_ridge_w0.5",
    ]
    return case_df.loc[case_df["variant_name"].isin(keep_variants)].copy().reset_index(drop=True)


def build_gate_lookup(try047) -> pd.DataFrame:
    final_records = [record for record in try047.scan_dataset_records() if record.is_labeled]
    added_records = try047.load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = try047.get_common_signal_columns(all_records)
    gate_df = try047.build_gate_feature_table(all_records, common_signal_columns)
    return gate_df.set_index("case_id")


def run_outer_fold(
    *,
    try047,
    try058,
    try053,
    train_records,
    holdout,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    gate_lookup: pd.DataFrame,
    domain_name: str,
    outer_model_dir: Path,
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

    train_feature_rows: list[dict[str, object]] = []
    for _, target_row in train_case_df.iterrows():
        candidate_df = train_case_df.loc[train_case_df["case_id"] != int(target_row["case_id"])].reset_index(drop=True)
        feature_row, _ = try058.build_case_feature_row(
            target_row=target_row,
            candidate_df=candidate_df,
            base_pred_column="rpm_pred_oof",
            residual_target_column="rpm_residual_oof",
            embed_mean=embed_mean,
            embed_std=embed_std,
            top_k=top_k,
            router_tag="train_case",
        )
        train_feature_rows.append(feature_row)

    holdout_feature_row, _ = try058.build_case_feature_row(
        target_row=holdout_row.iloc[0],
        candidate_df=train_case_df,
        base_pred_column="base_pred",
        residual_target_column=None,
        embed_mean=embed_mean,
        embed_std=embed_std,
        top_k=top_k,
        router_tag="holdout_case",
    )

    train_feature_df = pd.DataFrame(train_feature_rows).sort_values("case_id").reset_index(drop=True)
    holdout_feature_df = pd.DataFrame([holdout_feature_row])
    delta_feature_columns = try058.get_delta_only_feature_columns(train_feature_df)

    candidate_variant = "rpm_knn4__plus__delta_only_prototype_ridge_w0.5"
    holdout_candidate_pred = try058.predict_with_bounded_head(
        train_feature_df=train_feature_df,
        holdout_feature_df=holdout_feature_df,
        feature_columns=delta_feature_columns,
        model_name=f"{candidate_variant}__outer",
        model_dir=outer_model_dir,
        force_retrain=force_retrain,
        shrink=0.5,
    )

    train_gate_rows: list[dict[str, object]] = []
    for _, inner_test_row in train_feature_df.iterrows():
        inner_train_df = train_feature_df.loc[train_feature_df["case_id"] != int(inner_test_row["case_id"])].reset_index(drop=True)
        inner_test_df = pd.DataFrame([inner_test_row.to_dict()])
        candidate_pred = try058.predict_with_bounded_head(
            train_feature_df=inner_train_df,
            holdout_feature_df=inner_test_df,
            feature_columns=delta_feature_columns,
            model_name=f"{candidate_variant}__inner_outer{holdout.case_id}",
            model_dir=outer_model_dir,
            force_retrain=force_retrain,
            shrink=0.5,
        )
        gate_row = build_gate_row(
            feature_row=inner_test_row,
            candidate_pred=candidate_pred,
            gate_lookup=gate_lookup,
            true_wind_speed=float(inner_test_row["true_wind_speed"]),
            domain_name=("final_focus" if int(inner_test_row["case_id"]) < 21 else "added_focus"),
            row_role="train",
            holdout_case_id=holdout.case_id,
        )
        train_gate_rows.append(gate_row)

    holdout_gate_row = build_gate_row(
        feature_row=holdout_feature_df.iloc[0],
        candidate_pred=holdout_candidate_pred,
        gate_lookup=gate_lookup,
        true_wind_speed=float(holdout.wind_speed),
        domain_name=domain_name,
        row_role="holdout",
        holdout_case_id=holdout.case_id,
    )

    train_gate_df = pd.DataFrame(train_gate_rows).sort_values("case_id").reset_index(drop=True)
    holdout_gate_df = pd.DataFrame([holdout_gate_row])
    feature_columns = get_gate_feature_columns()

    case_rows: list[dict[str, object]] = []
    case_rows.append(build_case_row(domain_name, "delta_only_soft_gate_hgb", holdout, predict_soft_gate_hgb(train_gate_df, holdout_gate_df, feature_columns)))
    case_rows.append(
        build_case_row(
            domain_name,
            f"delta_only_binary_hgb_t{ENABLE_THRESHOLD:.2f}",
            holdout,
            predict_binary_gate(train_gate_df, holdout_gate_df, feature_columns, threshold=ENABLE_THRESHOLD),
        )
    )
    case_rows.append(build_case_row(domain_name, "delta_only_bucket_hgb", holdout, predict_bucket_gate(train_gate_df, holdout_gate_df, feature_columns)))
    case_rows.append(
        build_case_row(
            domain_name,
            f"delta_only_two_stage_hgb_t{ENABLE_THRESHOLD:.2f}",
            holdout,
            predict_two_stage_gate(train_gate_df, holdout_gate_df, feature_columns, threshold=ENABLE_THRESHOLD),
        )
    )
    case_rows.append(build_case_row(domain_name, "delta_only_trigger_rule_cv", holdout, predict_trigger_rule(train_gate_df, holdout_gate_df)))

    return case_rows, train_gate_rows, [*train_gate_rows, holdout_gate_row]


def build_gate_row(
    *,
    feature_row: pd.Series,
    candidate_pred: float,
    gate_lookup: pd.DataFrame,
    true_wind_speed: float,
    domain_name: str,
    row_role: str,
    holdout_case_id: int,
) -> dict[str, object]:
    case_id = int(feature_row["case_id"])
    gate_series = gate_lookup.loc[case_id]
    delta_values = np.asarray(
        [float(feature_row[column]) for column in feature_row.index if column.startswith("delta_embed_") and not column.startswith("abs_")],
        dtype=float,
    )
    abs_delta_values = np.abs(delta_values)
    base_pred = float(feature_row["base_pred"])
    pred_gap = float(candidate_pred - base_pred)
    optimal_gate = compute_optimal_gate_target(true_wind_speed=true_wind_speed, pred_base=base_pred, pred_enhanced=candidate_pred)
    bucket_label = nearest_bucket(optimal_gate)
    return {
        "holdout_case_id": int(holdout_case_id),
        "row_role": row_role,
        "domain": domain_name,
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
        "positive_bucket_label": float(map_positive_bucket(bucket_label)),
        "positive_bucket_class": int(positive_bucket_to_class(map_positive_bucket(bucket_label))),
    }


def compute_optimal_gate_target(*, true_wind_speed: float, pred_base: float, pred_enhanced: float) -> float:
    denom = float(pred_enhanced - pred_base)
    if abs(denom) <= EPS:
        return 0.0
    weight = (float(true_wind_speed) - float(pred_base)) / denom
    return float(np.clip(weight, 0.0, 1.0))


def nearest_bucket(value: float) -> float:
    return float(WEIGHT_BUCKETS[np.argmin(np.abs(WEIGHT_BUCKETS - value))])


def bucket_to_class(bucket_value: float) -> int:
    lookup = {0.0: 0, 0.3: 1, 0.5: 2, 1.0: 3}
    return lookup[float(bucket_value)]


def class_to_bucket(class_id: int) -> float:
    lookup = {0: 0.0, 1: 0.3, 2: 0.5, 3: 1.0}
    return lookup[int(class_id)]


def map_positive_bucket(bucket_value: float) -> float:
    if float(bucket_value) <= 0.0:
        return 0.3
    return float(bucket_value)


def positive_bucket_to_class(bucket_value: float) -> int:
    lookup = {0.3: 0, 0.5: 1, 1.0: 2}
    return lookup[float(bucket_value)]


def positive_class_to_bucket(class_id: int) -> float:
    lookup = {0: 0.3, 1: 0.5, 2: 1.0}
    return lookup[int(class_id)]


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
        min_samples_leaf=3,
        l2_regularization=0.1,
        random_state=42,
    )
    estimator.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["optimal_gate_target"].to_numpy(dtype=float))
    gate = float(np.clip(estimator.predict(test_df[feature_columns].to_numpy(dtype=float))[0], 0.0, 1.0))
    return blend_prediction(test_df, gate)


def predict_binary_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], threshold: float) -> float:
    estimator = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=3,
        l2_regularization=0.1,
        random_state=42,
    )
    estimator.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["enable_enhanced"].to_numpy(dtype=int))
    prob = float(estimator.predict_proba(test_df[feature_columns].to_numpy(dtype=float))[0, 1])
    gate = 1.0 if prob >= threshold else 0.0
    return blend_prediction(test_df, gate)


def predict_bucket_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> float:
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=0.5, max_iter=1000)),
        ]
    )
    estimator.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["bucket_class"].to_numpy(dtype=int))
    pred_class = int(estimator.predict(test_df[feature_columns].to_numpy(dtype=float))[0])
    gate = class_to_bucket(pred_class)
    return blend_prediction(test_df, gate)


def predict_two_stage_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], threshold: float) -> float:
    stage1 = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=3,
        l2_regularization=0.1,
        random_state=42,
    )
    stage1.fit(train_df[feature_columns].to_numpy(dtype=float), train_df["enable_enhanced"].to_numpy(dtype=int))
    enable_prob = float(stage1.predict_proba(test_df[feature_columns].to_numpy(dtype=float))[0, 1])
    if enable_prob < threshold:
        return blend_prediction(test_df, 0.0)

    positive_train_df = train_df.loc[train_df["enable_enhanced"] == 1].copy()
    if positive_train_df.empty:
        return blend_prediction(test_df, 0.0)
    stage2 = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=0.5, max_iter=1000)),
        ]
    )
    stage2.fit(positive_train_df[feature_columns].to_numpy(dtype=float), positive_train_df["positive_bucket_class"].to_numpy(dtype=int))
    positive_class = int(stage2.predict(test_df[feature_columns].to_numpy(dtype=float))[0])
    gate = positive_class_to_bucket(positive_class)
    return blend_prediction(test_df, gate)


def predict_trigger_rule(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    positive_gap_grid = [0.15, 0.20, 0.25]
    positive_top1_grid = [1.5, 2.0, 2.5]
    negative_gap_grid = [0.15, 0.20, 0.25]
    negative_std_grid = [1.5, 2.0, 2.5]
    strain_grid = [0.2, 1.0, 5.0]

    best_score = float("inf")
    best_params: tuple[float, float, float, float, float] | None = None
    for pos_gap in positive_gap_grid:
        for pos_top1 in positive_top1_grid:
            for neg_gap in negative_gap_grid:
                for neg_std in negative_std_grid:
                    for strain_limit in strain_grid:
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
        if abs_gap >= positive_gap and float(row["top1_embed_distance"]) <= positive_top1 and strain <= strain_limit:
            return 1.0
        return 0.0
    if pred_gap < 0.0:
        if abs_gap >= negative_gap and float(row["topk_embed_std_distance"]) <= negative_std and strain <= strain_limit:
            return 1.0
        return 0.0
    return 0.0


def blend_prediction(test_df: pd.DataFrame, gate: float) -> float:
    base_pred = float(test_df["base_pred"].iloc[0])
    candidate_pred = float(test_df["candidate_pred"].iloc[0])
    return float((1.0 - gate) * base_pred + gate * candidate_pred)


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
        "# delta-only gate / bucket / trigger quickcheck",
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

    focus_variants = [
        "rpm_knn4",
        "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
        "rpm_knn4__plus__delta_only_prototype_ridge_w0.5",
        "delta_only_soft_gate_hgb",
        f"delta_only_binary_hgb_t{ENABLE_THRESHOLD:.2f}",
        "delta_only_bucket_hgb",
        f"delta_only_two_stage_hgb_t{ENABLE_THRESHOLD:.2f}",
        "delta_only_trigger_rule_cv",
    ]
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
