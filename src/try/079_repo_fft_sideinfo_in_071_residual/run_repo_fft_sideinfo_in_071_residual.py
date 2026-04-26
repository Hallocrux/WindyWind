from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "079_repo_fft_sideinfo_in_071_residual"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
EMBEDDING_TABLE_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_case_table.csv"
FFT_BLEND_SCRIPT_PATH = (
    REPO_ROOT / "src" / "try" / "078_repo_fft_rpm_blend_quickcheck" / "run_repo_fft_rpm_blend_quickcheck.py"
)
FFT_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "043_1_fft_rpm_algorithm_search" / "run_fft_rpm_algorithm_search.py"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
K_NEIGHBORS = 4
EPS = 1e-6
SOURCE_COLUMN_ORDER = ["fft_peak_1x_whole", "window_peak_1x_conf_8s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 repo FFT side-info 并入 071 residual ridge 的 added-first quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--include-delta-only",
        action="store_true",
        help="额外加入 very small ablation：只在 embedding 后追加 repo_delta_rpm。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    external_df, embedding_columns, feature_columns_map = build_external_feature_table()
    variant_feature_columns = {
        "rpm_knn4_plus_embedding_residual_ridge": embedding_columns,
        "rpm_knn4_plus_embedding_repo_fft_sideinfo_residual_ridge": feature_columns_map["full_sideinfo"],
    }
    if args.include_delta_only:
        variant_feature_columns["rpm_knn4_plus_embedding_repo_delta_residual_ridge"] = feature_columns_map["delta_only"]

    loocv_pred_df = run_external_loocv(external_df, variant_feature_columns)
    transfer_pred_df = run_added_to_added2_transfer(external_df, variant_feature_columns)
    all_pred_df = pd.concat([loocv_pred_df, transfer_pred_df], ignore_index=True)

    summary_by_protocol_df = build_summary_by_protocol(all_pred_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(all_pred_df)
    transfer_compare_df = build_transfer_compare_table(
        pred_df=all_pred_df.loc[all_pred_df["protocol"] == "added_to_added2"].copy(),
        baseline_variant="rpm_knn4_plus_embedding_residual_ridge",
    )

    external_df.to_csv(output_dir / "external_feature_table.csv", index=False, encoding="utf-8-sig")
    build_feature_set_table(feature_columns_map, embedding_columns).to_csv(
        output_dir / "variant_feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_pred_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(output_dir / "summary_by_protocol_and_domain.csv", index=False, encoding="utf-8-sig")
    transfer_compare_df.to_csv(output_dir / "added_to_added2_compare_vs_071.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        summary_by_protocol_df=summary_by_protocol_df,
        summary_by_protocol_and_domain_df=summary_by_protocol_and_domain_df,
        transfer_compare_df=transfer_compare_df,
    )

    best_row = summary_by_protocol_df.iloc[0]
    print("079 repo fft side-info in 071 residual quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best protocol={best_row['protocol']} | "
        f"variant={best_row['variant_name']} | "
        f"case_mae={best_row['case_mae']:.4f}"
    )


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_external_feature_table() -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    embedding_df = pd.read_csv(EMBEDDING_TABLE_PATH, encoding="utf-8-sig")
    external_df = (
        embedding_df.loc[embedding_df["raw_source_domain"].isin(["added", "added2"]) & embedding_df["is_labeled"]]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    embedding_columns = [column for column in external_df.columns if column.startswith("embedding_")]

    fft_blend_module = load_module("repo_fft_blend_079", FFT_BLEND_SCRIPT_PATH)
    fft_module = load_module("repo_fft_core_079", FFT_SCRIPT_PATH)
    fft_case_df = fft_blend_module.build_case_table(fft_module)
    fft_feature_df = fft_blend_module.attach_repo_fft_features(fft_case_df, fft_module)

    merged_df = external_df.merge(
        fft_feature_df[
            [
                "case_id",
                "repo_fft_rpm",
                "repo_delta_rpm",
                "repo_abs_delta_rpm",
                "repo_fft_confidence",
                "repo_fft_source",
            ]
        ],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if merged_df["repo_fft_rpm"].isna().any():
        missing_case_ids = merged_df.loc[merged_df["repo_fft_rpm"].isna(), "case_id"].tolist()
        raise ValueError(f"FFT side-info 缺失 case_id: {missing_case_ids}")

    source_feature_columns: list[str] = []
    for source_name in SOURCE_COLUMN_ORDER:
        column_name = f"repo_fft_source__{source_name}"
        merged_df[column_name] = (merged_df["repo_fft_source"] == source_name).astype(float)
        source_feature_columns.append(column_name)

    feature_columns_map = {
        "baseline": embedding_columns,
        "delta_only": [*embedding_columns, "repo_delta_rpm"],
        "full_sideinfo": [
            *embedding_columns,
            "repo_fft_rpm",
            "repo_delta_rpm",
            "repo_abs_delta_rpm",
            "repo_fft_confidence",
            *source_feature_columns,
        ],
    }
    return merged_df, embedding_columns, feature_columns_map


def build_feature_set_table(feature_columns_map: dict[str, list[str]], embedding_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, feature_columns in feature_columns_map.items():
        extra_columns = [column for column in feature_columns if column not in embedding_columns]
        rows.append(
            {
                "variant_name": variant_name,
                "feature_count": int(len(feature_columns)),
                "embedding_feature_count": int(sum(column in embedding_columns for column in feature_columns)),
                "extra_feature_count": int(len(extra_columns)),
                "extra_feature_columns": ",".join(extra_columns),
            }
        )
    return pd.DataFrame(rows)


def run_external_loocv(external_df: pd.DataFrame, variant_feature_columns: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout_case_id in external_df["case_id"].tolist():
        train_df = external_df.loc[external_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_df = external_df.loc[external_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        rows.extend(build_prediction_rows("external_loocv", train_df, test_df, variant_feature_columns))
    return pd.DataFrame(rows)


def run_added_to_added2_transfer(external_df: pd.DataFrame, variant_feature_columns: dict[str, list[str]]) -> pd.DataFrame:
    train_df = external_df.loc[external_df["raw_source_domain"] == "added"].copy().reset_index(drop=True)
    test_df = external_df.loc[external_df["raw_source_domain"] == "added2"].copy().reset_index(drop=True)
    return pd.DataFrame(build_prediction_rows("added_to_added2", train_df, test_df, variant_feature_columns))


def build_prediction_rows(
    protocol: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant_feature_columns: dict[str, list[str]],
) -> list[dict[str, object]]:
    residual_targets = compute_internal_oof_residual_targets(train_df)
    base_test_pred = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))

    pred_map: dict[str, np.ndarray] = {"rpm_knn4": base_test_pred}
    for variant_name, feature_columns in variant_feature_columns.items():
        pred_map[variant_name] = base_test_pred + fit_predict_residual_ridge(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            residual_targets=residual_targets,
        )

    rows: list[dict[str, object]] = []
    for variant_name, pred_values in pred_map.items():
        pred_values = np.asarray(pred_values, dtype=float)
        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            pred_value = float(pred_values[row_idx])
            signed_error = pred_value - float(row["wind_speed"])
            rows.append(
                {
                    "protocol": protocol,
                    "variant_name": variant_name,
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["raw_source_domain"]),
                    "true_wind_speed": float(row["wind_speed"]),
                    "rpm": float(row["rpm"]),
                    "repo_fft_rpm": float(row["repo_fft_rpm"]),
                    "repo_delta_rpm": float(row["repo_delta_rpm"]),
                    "repo_abs_delta_rpm": float(row["repo_abs_delta_rpm"]),
                    "repo_fft_confidence": float(row["repo_fft_confidence"]),
                    "repo_fft_source": str(row["repo_fft_source"]),
                    "pred_wind_speed": pred_value,
                    "signed_error": signed_error,
                    "abs_error": abs(signed_error),
                }
            )
    return rows


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[float] = []
    for rpm_value in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm_value)
        order = np.argsort(distances)
        k = min(K_NEIGHBORS, len(order))
        order = order[:k]
        weights = 1.0 / np.maximum(distances[order], EPS)
        weights = weights / weights.sum()
        predictions.append(float(np.dot(weights, train_wind[order])))
    return np.asarray(predictions, dtype=float)


def compute_internal_oof_residual_targets(train_df: pd.DataFrame) -> np.ndarray:
    if len(train_df) <= 1:
        return np.zeros(len(train_df), dtype=float)
    rpm_values = train_df["rpm"].to_numpy(dtype=float, copy=False)
    true_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    residuals: list[float] = []
    for row_idx in range(len(train_df)):
        inner_train = train_df.drop(index=train_df.index[row_idx]).reset_index(drop=True)
        pred = compute_rpm_knn_predictions(inner_train, np.asarray([rpm_values[row_idx]], dtype=float))[0]
        residuals.append(float(true_wind[row_idx] - pred))
    return np.asarray(residuals, dtype=float)


def fit_predict_residual_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    residual_targets: np.ndarray,
) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    model.fit(
        train_df[feature_columns].to_numpy(dtype=float),
        np.asarray(residual_targets, dtype=float),
    )
    return model.predict(test_df[feature_columns].to_numpy(dtype=float))


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


def build_summary_by_protocol(all_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, variant_name), block in all_pred_df.groupby(["protocol", "variant_name"], sort=False):
        row = {"protocol": protocol, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_by_protocol_and_domain(all_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, domain, variant_name), block in all_pred_df.groupby(["protocol", "domain", "variant_name"], sort=False):
        row = {"protocol": protocol, "domain": domain, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_transfer_compare_table(pred_df: pd.DataFrame, baseline_variant: str) -> pd.DataFrame:
    pivot_df = pred_df.pivot_table(
        index=["case_id", "file_name", "domain", "true_wind_speed", "rpm", "repo_fft_rpm", "repo_delta_rpm", "repo_abs_delta_rpm", "repo_fft_confidence", "repo_fft_source"],
        columns="variant_name",
        values=["pred_wind_speed", "abs_error", "signed_error"],
    )
    pivot_df.columns = [
        f"{metric}__{variant_name}"
        for metric, variant_name in pivot_df.columns.to_flat_index()
    ]
    compare_df = pivot_df.reset_index().sort_values("case_id").reset_index(drop=True)

    baseline_error_column = f"abs_error__{baseline_variant}"
    if baseline_error_column not in compare_df.columns:
        return compare_df

    for variant_name in pred_df["variant_name"].unique():
        if variant_name == baseline_variant:
            continue
        candidate_column = f"abs_error__{variant_name}"
        if candidate_column in compare_df.columns:
            compare_df[f"abs_error_delta_vs_{baseline_variant}__{variant_name}"] = (
                compare_df[candidate_column] - compare_df[baseline_error_column]
            )
    return compare_df


def write_summary_markdown(
    output_path: Path,
    summary_by_protocol_df: pd.DataFrame,
    summary_by_protocol_and_domain_df: pd.DataFrame,
    transfer_compare_df: pd.DataFrame,
) -> None:
    lines = [f"# {TRY_NAME}", "", "## Summary By Protocol", ""]
    for protocol, protocol_df in summary_by_protocol_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in protocol_df.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`, "
                f"max_abs_error=`{row['max_abs_error']:.4f}`"
            )
        lines.append("")
    lines.extend(["## Summary By Protocol And Domain", ""])
    for (protocol, domain), block in summary_by_protocol_and_domain_df.groupby(["protocol", "domain"], sort=False):
        lines.append(f"### {protocol} | {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    if not transfer_compare_df.empty:
        lines.extend(["## Added To Added2 Case Compare", ""])
        for _, row in transfer_compare_df.iterrows():
            lines.append(
                f"- `case {int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, "
                f"rpm=`{row['rpm']:.4f}`, repo_fft_rpm=`{row['repo_fft_rpm']:.4f}`, "
                f"repo_delta_rpm=`{row['repo_delta_rpm']:+.4f}`, source=`{row['repo_fft_source']}`, "
                f"confidence=`{row['repo_fft_confidence']:.4f}`"
            )
            for column in sorted(column for column in transfer_compare_df.columns if column.startswith("abs_error__")):
                variant_name = column.split("__", 1)[1]
                pred_column = f"pred_wind_speed__{variant_name}"
                signed_column = f"signed_error__{variant_name}"
                lines.append(
                    f"  - `{variant_name}`: pred=`{row[pred_column]:.4f}`, "
                    f"signed_error=`{row[signed_column]:+.4f}`, abs_error=`{row[column]:.4f}`"
                )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
