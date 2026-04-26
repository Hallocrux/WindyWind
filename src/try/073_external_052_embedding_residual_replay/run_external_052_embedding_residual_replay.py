from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "073_external_052_embedding_residual_replay"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
EMBEDDING_TABLE_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_case_table.csv"
RPM_K = 4
EMBED_K = 4
SHRINK = 0.5
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 external-first 口径下 replay 052 embedding residual。")
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

    loocv_pred_df, loocv_neighbor_df = run_external_loocv(external_df, embedding_columns)
    transfer_pred_df, transfer_neighbor_df = run_added_to_added2_transfer(external_df, embedding_columns)
    all_pred_df = pd.concat([loocv_pred_df, transfer_pred_df], ignore_index=True)
    neighbor_df = pd.concat([loocv_neighbor_df, transfer_neighbor_df], ignore_index=True)

    summary_by_protocol_df = build_summary_by_protocol(all_pred_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(all_pred_df)

    all_pred_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(
        output_dir / "summary_by_protocol_and_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    neighbor_df.to_csv(output_dir / "neighbor_table.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_by_protocol_df, summary_by_protocol_and_domain_df)

    best_row = summary_by_protocol_df.iloc[0]
    print("073 external 052 embedding residual replay 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best protocol={best_row['protocol']} | "
        f"variant={best_row['variant_name']} | "
        f"case_mae={best_row['case_mae']:.4f}"
    )


def run_external_loocv(
    external_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for holdout_case_id in external_df["case_id"].tolist():
        train_df = external_df.loc[external_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_df = external_df.loc[external_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        fold_case_rows, fold_neighbor_rows = build_prediction_rows(
            protocol="external_loocv",
            train_df=train_df,
            test_df=test_df,
            embedding_columns=embedding_columns,
        )
        case_rows.extend(fold_case_rows)
        neighbor_rows.extend(fold_neighbor_rows)
    return pd.DataFrame(case_rows), pd.DataFrame(neighbor_rows)


def run_added_to_added2_transfer(
    external_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = external_df.loc[external_df["raw_source_domain"] == "added"].copy().reset_index(drop=True)
    test_df = external_df.loc[external_df["raw_source_domain"] == "added2"].copy().reset_index(drop=True)
    case_rows, neighbor_rows = build_prediction_rows(
        protocol="added_to_added2",
        train_df=train_df,
        test_df=test_df,
        embedding_columns=embedding_columns,
    )
    return pd.DataFrame(case_rows), pd.DataFrame(neighbor_rows)


def build_prediction_rows(
    *,
    protocol: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    residual_targets = compute_internal_oof_residual_targets(train_df)
    base_preds = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))
    replay_preds, neighbor_rows = compute_052_residual_predictions(
        train_df=train_df,
        test_df=test_df,
        embedding_columns=embedding_columns,
        residual_targets=residual_targets,
        protocol=protocol,
    )

    rows: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        pred_map = {
            "rpm_knn4": float(base_preds[idx]),
            "052_embedding_residual_knn4_concat_2s_8s_w0.5": float(base_preds[idx] + SHRINK * replay_preds[idx]),
        }
        for variant_name, pred_value in pred_map.items():
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
                    "pred_wind_speed": pred_value,
                    "signed_error": signed_error,
                    "abs_error": abs(signed_error),
                }
            )
    return rows, neighbor_rows


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


def compute_internal_oof_residual_targets(train_df: pd.DataFrame) -> np.ndarray:
    if len(train_df) <= 1:
        return np.zeros(len(train_df), dtype=float)
    residuals: list[float] = []
    rpm_values = train_df["rpm"].to_numpy(dtype=float, copy=False)
    true_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    for idx in range(len(train_df)):
        inner_train_df = train_df.drop(index=train_df.index[idx]).reset_index(drop=True)
        pred = compute_rpm_knn_predictions(inner_train_df, np.asarray([rpm_values[idx]], dtype=float))[0]
        residuals.append(float(true_wind[idx] - pred))
    return np.asarray(residuals, dtype=float)


def compute_052_residual_predictions(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
    residual_targets: np.ndarray,
    protocol: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    train_matrix = train_df[embedding_columns].to_numpy(dtype=float, copy=False)
    test_matrix = test_df[embedding_columns].to_numpy(dtype=float, copy=False)
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    train_scaled = (train_matrix - mean) / std
    test_scaled = (test_matrix - mean) / std

    predictions: list[float] = []
    neighbor_rows: list[dict[str, object]] = []
    for test_idx, (_, row) in enumerate(test_df.iterrows()):
        distances = np.linalg.norm(train_scaled - test_scaled[test_idx], axis=1)
        order = np.argsort(distances)[: min(EMBED_K, len(train_df))]
        weights = 1.0 / np.maximum(distances[order], EPS)
        weights = weights / weights.sum()
        residual_pred = float(np.dot(weights, residual_targets[order]))
        predictions.append(residual_pred)
        for rank, train_idx in enumerate(order, start=1):
            neighbor_rows.append(
                {
                    "protocol": protocol,
                    "holdout_case_id": int(row["case_id"]),
                    "holdout_domain": str(row["raw_source_domain"]),
                    "variant_name": "052_embedding_residual_knn4_concat_2s_8s_w0.5",
                    "neighbor_rank": rank,
                    "neighbor_case_id": int(train_df.iloc[train_idx]["case_id"]),
                    "neighbor_domain": str(train_df.iloc[train_idx]["raw_source_domain"]),
                    "distance": float(distances[train_idx]),
                    "weight": float(weights[rank - 1]),
                    "neighbor_rpm_residual_oof": float(residual_targets[train_idx]),
                }
            )
    return np.asarray(predictions, dtype=float), neighbor_rows


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
    for (protocol, domain, variant_name), block in all_pred_df.groupby(
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
    lines = ["# external 052 embedding residual replay", "", "## Summary By Protocol", ""]
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
