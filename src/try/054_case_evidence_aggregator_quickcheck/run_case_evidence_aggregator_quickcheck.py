from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "054_case_evidence_aggregator_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY052_CASE_PATH = REPO_ROOT / "outputs" / "try" / "052_tcn_embedding_window_signal_quickcheck" / "case_level_predictions.csv"
TRY053_CASE_PATH = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "case_level_predictions.csv"
TRY047_GATE_FEATURE_PATH = REPO_ROOT / "outputs" / "try" / "047_soft_gate_quickcheck" / "gate_feature_table.csv"
HOLDOUT_CASE_IDS = [1, 3, 17, 18, 21, 22, 23, 24]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
BASELINE_VARIANTS = [
    "rpm_knn4",
    "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
    "rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="case evidence aggregator quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    evidence_df = build_evidence_table()
    evidence_df.to_csv(output_dir / "evidence_table.csv", index=False, encoding="utf-8-sig")

    case_rows: list[dict[str, object]] = []
    for _, test_row in evidence_df.iterrows():
        train_df = evidence_df.loc[evidence_df["case_id"] != int(test_row["case_id"])].reset_index(drop=True)
        feature_sets = {
            "ridge_candidate_only": get_candidate_feature_columns(),
            "ridge_candidate_plus_mechanism": [*get_candidate_feature_columns(), *get_mechanism_feature_columns()],
            "ridge_consensus_plus_mechanism": [*get_consensus_feature_columns(), *get_mechanism_feature_columns()],
        }
        base_pred = float(test_row["base_pred"])
        case_rows.append(build_case_row(test_row, "rpm_knn4", base_pred))
        for variant_name in BASELINE_VARIANTS[1:]:
            case_rows.append(build_case_row(test_row, variant_name, float(test_row[f"pred::{variant_name}"])))
        for model_name, feature_columns in feature_sets.items():
            pred = predict_with_fold_model(
                train_df=train_df,
                test_row=test_row,
                feature_columns=feature_columns,
                model_name=model_name,
                model_dir=model_dir,
                force_retrain=args.force_retrain,
            )
            case_rows.append(build_case_row(test_row, model_name, pred))
            case_rows.append(build_case_row(test_row, f"{model_name}__w0.5", base_pred + 0.5 * (pred - base_pred)))

    case_level_df = pd.DataFrame(case_rows)
    summary_df = build_summary_by_domain(case_level_df)
    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    write_summary(output_dir / "summary.md", summary_df, case_level_df)

    best_focus = summary_df.loc[summary_df["domain"] == "focus_all"].iloc[0]
    print("054 case evidence aggregator quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best focus_all: {best_focus['variant_name']} | case_mae={best_focus['case_mae']:.4f}")


def build_evidence_table() -> pd.DataFrame:
    case_052 = pd.read_csv(TRY052_CASE_PATH, encoding="utf-8-sig")
    case_053 = pd.read_csv(TRY053_CASE_PATH, encoding="utf-8-sig")
    gate_df = pd.read_csv(TRY047_GATE_FEATURE_PATH, encoding="utf-8-sig")

    focus_052 = case_052.loc[case_052["case_id"].isin(HOLDOUT_CASE_IDS)].copy()
    focus_053 = case_053.loc[case_053["case_id"].isin(HOLDOUT_CASE_IDS)].copy()
    focus_gate = gate_df.loc[gate_df["case_id"].isin(HOLDOUT_CASE_IDS)].copy()

    base_df = focus_052.loc[focus_052["variant_name"] == "rpm_knn4", ["case_id", "file_name", "domain", "true_wind_speed", "rpm", "pred_wind_speed"]].copy()
    base_df = base_df.rename(columns={"pred_wind_speed": "base_pred", "rpm": "true_rpm"})

    variant_frames: dict[str, pd.DataFrame] = {}
    for source_df in (focus_052, focus_053):
        for variant_name in sorted(source_df["variant_name"].unique()):
            if variant_name == "rpm_knn4" or variant_name in variant_frames:
                continue
            block = source_df.loc[source_df["variant_name"] == variant_name, ["case_id", "pred_wind_speed"]].copy()
            block = block.rename(columns={"pred_wind_speed": f"pred::{variant_name}"})
            variant_frames[variant_name] = block

    merged = base_df.copy()
    for frame in variant_frames.values():
        merged = merged.merge(frame, on="case_id", how="left")

    merged = merged.merge(
        focus_gate[
            [
                "case_id",
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
        ],
        on="case_id",
        how="left",
    )

    merged["residual_target"] = merged["true_wind_speed"] - merged["base_pred"]

    candidate_variants = [
        "rpm_knn4__plus__embedding_residual_knn4_2s_w0.5",
        "rpm_knn4__plus__embedding_residual_knn4_5s_w0.5",
        "rpm_knn4__plus__embedding_residual_knn4_8s_w0.5",
        "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
        "rpm_knn4__plus__support_window_residual_2s_w0.5",
        "rpm_knn4__plus__support_window_residual_8s_w0.5",
        "rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5",
        "rpm_knn4__plus__support_window_residual_concat_2s_8s_w0.5",
    ]
    correction_columns: list[str] = []
    for variant_name in candidate_variants:
        pred_col = f"pred::{variant_name}"
        corr_col = f"corr::{variant_name}"
        merged[corr_col] = merged[pred_col] - merged["base_pred"]
        correction_columns.append(corr_col)

    correction_matrix = merged[correction_columns].to_numpy(dtype=float)
    merged["corr_mean"] = correction_matrix.mean(axis=1)
    merged["corr_std"] = correction_matrix.std(axis=1)
    merged["corr_min"] = correction_matrix.min(axis=1)
    merged["corr_max"] = correction_matrix.max(axis=1)
    merged["corr_abs_mean"] = np.abs(correction_matrix).mean(axis=1)
    merged["corr_abs_max"] = np.abs(correction_matrix).max(axis=1)
    merged["corr_pos_count"] = (correction_matrix > 0).sum(axis=1)
    merged["corr_neg_count"] = (correction_matrix < 0).sum(axis=1)
    merged["corr_embed_support_gap"] = (
        merged["corr::rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5"]
        - merged["corr::rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5"]
    )
    merged["corr_short_long_gap"] = (
        merged["corr::rpm_knn4__plus__embedding_residual_knn4_2s_w0.5"]
        - merged["corr::rpm_knn4__plus__embedding_residual_knn4_8s_w0.5"]
    )
    merged["base_abs_error"] = (merged["base_pred"] - merged["true_wind_speed"]).abs()
    return merged.sort_values("case_id").reset_index(drop=True)


def predict_with_fold_model(
    *,
    train_df: pd.DataFrame,
    test_row: pd.Series,
    feature_columns: list[str],
    model_name: str,
    model_dir: Path,
    force_retrain: bool,
) -> float:
    case_id = int(test_row["case_id"])
    model_path = model_dir / f"{model_name}_fold_case_{case_id}.pkl"
    meta_path = model_dir / f"{model_name}_fold_case_{case_id}.json"
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["residual_target"].to_numpy(dtype=float)
    X_test = test_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)

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
        model.fit(X_train, y_train)
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "holdout_case_id": case_id,
                    "feature_columns": feature_columns,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    pred_residual = float(model.predict(X_test)[0])
    clip_value = max(float(np.quantile(np.abs(y_train), 0.9)), 0.15)
    pred_residual = float(np.clip(pred_residual, -clip_value, clip_value))
    return float(test_row["base_pred"] + pred_residual)


def get_candidate_feature_columns() -> list[str]:
    return [
        "corr::rpm_knn4__plus__embedding_residual_knn4_2s_w0.5",
        "corr::rpm_knn4__plus__embedding_residual_knn4_5s_w0.5",
        "corr::rpm_knn4__plus__embedding_residual_knn4_8s_w0.5",
        "corr::rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
        "corr::rpm_knn4__plus__support_window_residual_2s_w0.5",
        "corr::rpm_knn4__plus__support_window_residual_8s_w0.5",
        "corr::rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5",
        "corr::rpm_knn4__plus__support_window_residual_concat_2s_8s_w0.5",
    ]


def get_consensus_feature_columns() -> list[str]:
    return [
        "corr_mean",
        "corr_std",
        "corr_min",
        "corr_max",
        "corr_abs_mean",
        "corr_abs_max",
        "corr_pos_count",
        "corr_neg_count",
        "corr_embed_support_gap",
        "corr_short_long_gap",
        "base_pred",
        "true_rpm",
        "base_abs_error",
    ]


def get_mechanism_feature_columns() -> list[str]:
    return [
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
        "true_rpm",
        "base_pred",
    ]


def build_case_row(test_row: pd.Series, variant_name: str, pred_wind_speed: float) -> dict[str, object]:
    signed_error = float(pred_wind_speed - float(test_row["true_wind_speed"]))
    return {
        "domain": str(test_row["domain"]),
        "variant_name": variant_name,
        "case_id": int(test_row["case_id"]),
        "file_name": str(test_row["file_name"]),
        "true_wind_speed": float(test_row["true_wind_speed"]),
        "rpm": float(test_row["true_rpm"]),
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


def write_summary(output_path: Path, summary_df: pd.DataFrame, case_level_df: pd.DataFrame) -> None:
    lines = [
        "# case evidence aggregator quickcheck",
        "",
        f"- holdout 工况：`{', '.join(str(case_id) for case_id in HOLDOUT_CASE_IDS)}`",
        "",
        "## 三桶汇总",
        "",
    ]
    for domain_name in ("final_focus", "added_focus", "focus_all"):
        lines.append(f"### {domain_name}")
        lines.append("")
        block = summary_df.loc[summary_df["domain"] == domain_name].copy()
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    lines.extend(["## 每工况重点对照", ""])
    for case_id, block in case_level_df.groupby("case_id", sort=True):
        lines.append(f"### 工况{int(case_id)}")
        lines.append("")
        for _, row in block.sort_values(["abs_error", "variant_name"]).iterrows():
            lines.append(
                f"- `{row['variant_name']}`: true=`{row['true_wind_speed']:.4f}`, pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
