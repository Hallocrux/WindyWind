from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRY_NAME = "048_conservative_gate_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY047_SCRIPT = REPO_ROOT / "src" / "try" / "047_soft_gate_quickcheck" / "run_soft_gate_quickcheck.py"
TRY047_CASE_PATH = REPO_ROOT / "outputs" / "try" / "047_soft_gate_quickcheck" / "case_level_predictions.csv"
WEIGHT_BUCKETS = np.array([0.0, 0.3, 0.5, 1.0], dtype=float)
POSITIVE_WEIGHT_BUCKETS = np.array([0.3, 0.5, 1.0], dtype=float)
ENABLE_THRESHOLD = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证更保守的 binary / bucketed / two-stage gate。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try047 = load_try047_module()
    dataset_df = build_dataset_table(try047)

    prediction_rows: list[dict[str, object]] = []
    for test_case_id in dataset_df["case_id"]:
        train_df = dataset_df.loc[dataset_df["case_id"] != test_case_id].copy()
        test_df = dataset_df.loc[dataset_df["case_id"] == test_case_id].copy()
        feature_columns = try047.get_gate_feature_columns()

        prediction_rows.extend(build_binary_predictions(train_df, test_df, feature_columns))
        prediction_rows.extend(build_bucket_predictions(train_df, test_df, feature_columns))
        prediction_rows.extend(build_two_stage_predictions(train_df, test_df, feature_columns))

    conservative_df = pd.DataFrame(prediction_rows)
    conservative_df["signed_error"] = conservative_df["pred_wind_speed"] - conservative_df["true_wind_speed"]
    conservative_df["abs_error"] = conservative_df["signed_error"].abs()

    baseline_df = build_baseline_rows(dataset_df)
    case_level_df = pd.concat([conservative_df, baseline_df], ignore_index=True)
    summary_by_variant_df = build_summary_by_variant(case_level_df)
    summary_by_domain_df = build_summary_by_domain(summary_by_variant_df)

    dataset_df.to_csv(output_dir / "dataset_table.csv", index=False, encoding="utf-8-sig")
    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_variant_df.to_csv(output_dir / "summary_by_variant.csv", index=False, encoding="utf-8-sig")
    summary_by_domain_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_by_variant_df, summary_by_domain_df)

    best_all = summary_by_variant_df.loc[summary_by_variant_df["domain"] == "all_labeled"].iloc[0]
    print("048 conservative gate quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best all_labeled: {best_all['variant_name']} | case_mae={best_all['case_mae']:.4f}")


def load_try047_module():
    spec = importlib.util.spec_from_file_location("try047_module", TRY047_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {TRY047_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["try047_module"] = module
    spec.loader.exec_module(module)
    return module


def build_dataset_table(try047) -> pd.DataFrame:
    final_records = [record for record in try047.scan_dataset_records() if record.is_labeled]
    added_records = try047.load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = try047.get_common_signal_columns(all_records)
    gate_feature_df = try047.build_gate_feature_table(all_records, common_signal_columns)
    expert_df = try047.build_expert_prediction_table()
    dataset_df = gate_feature_df.merge(
        expert_df,
        on=["case_id", "file_name", "true_wind_speed", "domain"],
        how="inner",
    )
    dataset_df["pred_gap"] = dataset_df["pred_enhanced"] - dataset_df["pred_base"]
    dataset_df["abs_pred_gap"] = dataset_df["pred_gap"].abs()
    dataset_df["optimal_gate_target"] = try047.compute_optimal_gate_target(
        true_values=dataset_df["true_wind_speed"].to_numpy(dtype=float),
        pred_base=dataset_df["pred_base"].to_numpy(dtype=float),
        pred_enhanced=dataset_df["pred_enhanced"].to_numpy(dtype=float),
    )
    dataset_df["enhanced_better"] = (
        (dataset_df["pred_enhanced"] - dataset_df["true_wind_speed"]).abs()
        < (dataset_df["pred_base"] - dataset_df["true_wind_speed"]).abs()
    ).astype(int)
    dataset_df["bucket_label"] = dataset_df["optimal_gate_target"].apply(nearest_bucket).astype(float)
    dataset_df["bucket_class"] = dataset_df["bucket_label"].map(bucket_to_class).astype(int)
    dataset_df["enable_enhanced"] = (dataset_df["bucket_label"] > 0.0).astype(int)
    dataset_df["positive_bucket_label"] = dataset_df["bucket_label"].apply(map_positive_bucket)
    dataset_df["positive_bucket_class"] = dataset_df["positive_bucket_label"].map(positive_bucket_to_class).astype(int)
    return dataset_df.sort_values("case_id").reset_index(drop=True)


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


def build_binary_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    models = {
        "binary_logit_hard": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=0.5, max_iter=1000)),
            ]
        ),
        "binary_hgb_hard": HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=3,
            l2_regularization=0.1,
            random_state=42,
        ),
    }
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["enable_enhanced"].to_numpy(dtype=int)
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    for variant_name, estimator in models.items():
        estimator.fit(X_train, y_train)
        prob = float(estimator.predict_proba(X_test)[0, 1])
        gate = 1.0 if prob >= 0.5 else 0.0
        rows.append(build_prediction_row(variant_name, test_df, gate, prob))

        conservative_gate = 1.0 if prob >= ENABLE_THRESHOLD else 0.0
        rows.append(build_prediction_row(f"{variant_name}_t{ENABLE_THRESHOLD:.2f}", test_df, conservative_gate, prob))
    return rows


def build_bucket_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    models = {
        "bucket_logit": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=0.5, max_iter=1000)),
            ]
        ),
        "bucket_hgb": HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=3,
            l2_regularization=0.1,
            random_state=42,
        ),
    }
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["bucket_class"].to_numpy(dtype=int)
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    for variant_name, estimator in models.items():
        estimator.fit(X_train, y_train)
        pred_class = int(estimator.predict(X_test)[0])
        probs = estimator.predict_proba(X_test)[0]
        confidence = float(np.max(probs))
        gate = class_to_bucket(pred_class)
        rows.append(build_prediction_row(variant_name, test_df, gate, confidence))
    return rows


def build_two_stage_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stage1_models = {
        "two_stage_logit": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=0.5, max_iter=1000)),
            ]
        ),
        "two_stage_hgb": HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=3,
            l2_regularization=0.1,
            random_state=42,
        ),
    }
    positive_train_df = train_df.loc[train_df["enable_enhanced"] == 1].copy()
    if positive_train_df.empty:
        return rows

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_enable = train_df["enable_enhanced"].to_numpy(dtype=int)
    X_test = test_df[feature_columns].to_numpy(dtype=float)

    X_positive = positive_train_df[feature_columns].to_numpy(dtype=float)
    y_positive = positive_train_df["positive_bucket_class"].to_numpy(dtype=int)

    for variant_name, stage1_estimator in stage1_models.items():
        stage1_estimator.fit(X_train, y_enable)
        enable_prob = float(stage1_estimator.predict_proba(X_test)[0, 1])
        stage2_estimator = build_stage2_estimator(variant_name)
        stage2_estimator.fit(X_positive, y_positive)
        positive_class = int(stage2_estimator.predict(X_test)[0])
        positive_weight = positive_class_to_bucket(positive_class)

        hard_gate = positive_weight if enable_prob >= 0.5 else 0.0
        rows.append(build_prediction_row(variant_name, test_df, hard_gate, enable_prob))

        conservative_gate = positive_weight if enable_prob >= ENABLE_THRESHOLD else 0.0
        rows.append(build_prediction_row(f"{variant_name}_t{ENABLE_THRESHOLD:.2f}", test_df, conservative_gate, enable_prob))

    return rows


def build_stage2_estimator(variant_name: str):
    if variant_name == "two_stage_logit":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=0.5, max_iter=1000)),
            ]
        )
    return HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=3,
        l2_regularization=0.1,
        random_state=42,
    )


def build_prediction_row(
    variant_name: str,
    test_df: pd.DataFrame,
    gate: float,
    auxiliary_score: float,
) -> dict[str, object]:
    pred_base = float(test_df["pred_base"].iloc[0])
    pred_enhanced = float(test_df["pred_enhanced"].iloc[0])
    pred_wind = (1.0 - gate) * pred_base + gate * pred_enhanced
    return {
        "variant_name": variant_name,
        "case_id": int(test_df["case_id"].iloc[0]),
        "file_name": str(test_df["file_name"].iloc[0]),
        "domain": str(test_df["domain"].iloc[0]),
        "true_wind_speed": float(test_df["true_wind_speed"].iloc[0]),
        "pred_base": pred_base,
        "pred_enhanced": pred_enhanced,
        "pred_gate": float(gate),
        "optimal_gate_target": float(test_df["optimal_gate_target"].iloc[0]),
        "auxiliary_score": auxiliary_score,
        "pred_wind_speed": float(pred_wind),
    }


def build_baseline_rows(dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in dataset_df.iterrows():
        rows.extend(
            [
                {
                    "variant_name": "base_only",
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["domain"]),
                    "true_wind_speed": float(row["true_wind_speed"]),
                    "pred_base": float(row["pred_base"]),
                    "pred_enhanced": float(row["pred_enhanced"]),
                    "pred_gate": 0.0,
                    "optimal_gate_target": float(row["optimal_gate_target"]),
                    "auxiliary_score": np.nan,
                    "pred_wind_speed": float(row["pred_base"]),
                },
                {
                    "variant_name": "enhanced_only",
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["domain"]),
                    "true_wind_speed": float(row["true_wind_speed"]),
                    "pred_base": float(row["pred_base"]),
                    "pred_enhanced": float(row["pred_enhanced"]),
                    "pred_gate": 1.0,
                    "optimal_gate_target": float(row["optimal_gate_target"]),
                    "auxiliary_score": np.nan,
                    "pred_wind_speed": float(row["pred_enhanced"]),
                },
                {
                    "variant_name": "oracle_soft_gate",
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["domain"]),
                    "true_wind_speed": float(row["true_wind_speed"]),
                    "pred_base": float(row["pred_base"]),
                    "pred_enhanced": float(row["pred_enhanced"]),
                    "pred_gate": float(row["optimal_gate_target"]),
                    "optimal_gate_target": float(row["optimal_gate_target"]),
                    "auxiliary_score": np.nan,
                    "pred_wind_speed": float(
                        (1.0 - float(row["optimal_gate_target"])) * row["pred_base"]
                        + float(row["optimal_gate_target"]) * row["pred_enhanced"]
                    ),
                },
            ]
        )

    ref_df = pd.read_csv(TRY047_CASE_PATH, encoding="utf-8-sig")
    ref_df = ref_df.loc[ref_df["variant_name"].isin(["global_weight_cv", "hgb_gate"])].copy()
    for variant_name in ("global_weight_cv", "hgb_gate"):
        block = ref_df.loc[ref_df["variant_name"] == variant_name].copy()
        block["auxiliary_score"] = np.nan
        rows.extend(block.to_dict(orient="records"))
    baseline_df = pd.DataFrame(rows)
    baseline_df["signed_error"] = baseline_df["pred_wind_speed"] - baseline_df["true_wind_speed"]
    baseline_df["abs_error"] = baseline_df["signed_error"].abs()
    return baseline_df


def build_summary_by_variant(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_level_df.groupby("variant_name", sort=False):
        rows.append(build_summary_row("all_labeled", variant_name, block))
        for domain_name, domain_block in block.groupby("domain", sort=False):
            rows.append(build_summary_row(domain_name, variant_name, domain_block))
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_row(domain: str, variant_name: str, block: pd.DataFrame) -> dict[str, object]:
    gate_values = block["pred_gate"].to_numpy(dtype=float)
    return {
        "domain": domain,
        "variant_name": variant_name,
        "case_mae": float(mean_absolute_error(block["true_wind_speed"], block["pred_wind_speed"])),
        "case_rmse": float(np.sqrt(mean_squared_error(block["true_wind_speed"], block["pred_wind_speed"]))),
        "mean_signed_error": float(block["signed_error"].mean()),
        "mean_gate": float(np.mean(gate_values)),
        "gate_std": float(np.std(gate_values)),
        "case_count": int(len(block)),
    }


def build_summary_by_domain(summary_by_variant_df: pd.DataFrame) -> pd.DataFrame:
    target_variants = [
        "base_only",
        "enhanced_only",
        "global_weight_cv",
        "hgb_gate",
        "binary_hgb_hard_t0.65",
        "bucket_hgb",
        "two_stage_hgb_t0.65",
        "oracle_soft_gate",
    ]
    rows: list[dict[str, object]] = []
    for domain, block in summary_by_variant_df.groupby("domain", sort=False):
        filtered = block.loc[block["variant_name"].isin(target_variants)].copy()
        best_row = filtered.iloc[0]
        rows.append(
            {
                "domain": domain,
                "best_variant": best_row["variant_name"],
                "best_case_mae": float(best_row["case_mae"]),
                "base_only_case_mae": float(block.loc[block["variant_name"] == "base_only", "case_mae"].iloc[0]),
                "enhanced_only_case_mae": float(block.loc[block["variant_name"] == "enhanced_only", "case_mae"].iloc[0]),
                "global_weight_cv_case_mae": float(block.loc[block["variant_name"] == "global_weight_cv", "case_mae"].iloc[0]),
                "hgb_gate_047_case_mae": float(block.loc[block["variant_name"] == "hgb_gate", "case_mae"].iloc[0]),
                "oracle_soft_gate_case_mae": float(block.loc[block["variant_name"] == "oracle_soft_gate", "case_mae"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(
    output_path: Path,
    summary_by_variant_df: pd.DataFrame,
    summary_by_domain_df: pd.DataFrame,
) -> None:
    focus_variants = [
        "base_only",
        "enhanced_only",
        "global_weight_cv",
        "hgb_gate",
        "binary_hgb_hard_t0.65",
        "bucket_hgb",
        "two_stage_hgb_t0.65",
        "oracle_soft_gate",
    ]
    lines = [
        f"# {TRY_NAME}",
        "",
        "## Domain Summary",
        "",
    ]
    for _, row in summary_by_domain_df.iterrows():
        lines.append(f"### {row['domain']}")
        lines.append("")
        lines.append(f"- best conservative variant: `{row['best_variant']}` | case_mae=`{row['best_case_mae']:.4f}`")
        lines.append(f"- `base_only`: case_mae=`{row['base_only_case_mae']:.4f}`")
        lines.append(f"- `enhanced_only`: case_mae=`{row['enhanced_only_case_mae']:.4f}`")
        lines.append(f"- `global_weight_cv`: case_mae=`{row['global_weight_cv_case_mae']:.4f}`")
        lines.append(f"- `047 hgb_gate`: case_mae=`{row['hgb_gate_047_case_mae']:.4f}`")
        lines.append(f"- `oracle_soft_gate`: case_mae=`{row['oracle_soft_gate_case_mae']:.4f}`")
        lines.append("")

    lines.extend(["## Focus Variants", ""])
    for domain, block in summary_by_variant_df.groupby("domain", sort=False):
        lines.append(f"### {domain}")
        lines.append("")
        focus_block = block.loc[block["variant_name"].isin(focus_variants)].copy()
        for _, row in focus_block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_gate=`{row['mean_gate']:.4f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
