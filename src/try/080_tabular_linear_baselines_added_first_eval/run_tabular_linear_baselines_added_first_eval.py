from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "080_tabular_linear_baselines_added_first_eval"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
REFERENCE_071_PRED_PATH = (
    REPO_ROOT / "outputs" / "try" / "071_external_embedding_regression_quickcheck" / "all_case_predictions.csv"
)
EPS = 1e-6
K_NEIGHBORS = 4

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRY009_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY009_ROOT))

from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame  # noqa: E402
from src.current.features import WindowConfig, build_case_feature_frame, get_vibration_feature_columns  # noqa: E402
from phase1_feature_groups_lib import build_feature_frame, get_group_feature_columns  # noqa: E402

ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补测 added-first 口径下的 G6 与线性表格基线。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_external_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }

    window_config = WindowConfig()
    g6_feature_df = build_feature_frame(records, cleaned_signal_frames, window_config)
    vib_ft_feature_df = build_vib_ft_feature_frame(records, cleaned_signal_frames, window_config)

    pred_frames = [
        run_tabular_protocols(
            feature_df=g6_feature_df,
            protocol_name="tabular_reference_g6_ridge",
            feature_columns=get_group_feature_columns(g6_feature_df, "G6_TIME_FREQ_CROSS"),
        ),
        run_tabular_protocols(
            feature_df=vib_ft_feature_df,
            protocol_name="ridge_vib_ft_rpm",
            feature_columns=[*get_vibration_feature_columns(vib_ft_feature_df), "rpm"],
        ),
    ]
    pred_frames.append(run_rpm_knn4_protocols(records))
    ref_071_df = load_071_reference_predictions()
    if ref_071_df is not None:
        pred_frames.append(ref_071_df)

    all_pred_df = pd.concat(pred_frames, ignore_index=True)
    summary_by_protocol_df = build_summary_by_protocol(all_pred_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(all_pred_df)

    g6_feature_df.to_csv(output_dir / "g6_feature_frame.csv", index=False, encoding="utf-8-sig")
    vib_ft_feature_df.to_csv(output_dir / "vib_ft_feature_frame.csv", index=False, encoding="utf-8-sig")
    all_pred_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(
        output_dir / "summary_by_protocol_and_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir / "summary.md", summary_by_protocol_df, summary_by_protocol_and_domain_df)

    best_row = summary_by_protocol_df.iloc[0]
    print("080 tabular linear baselines added-first eval 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best protocol={best_row['protocol']} | "
        f"variant={best_row['variant_name']} | "
        f"case_mae={best_row['case_mae']:.4f}"
    )


def build_external_records() -> list[DatasetRecord]:
    rows: list[DatasetRecord] = []
    for manifest_path, data_dir in (
        (ADDED_MANIFEST_PATH, ADDED_DIR),
        (ADDED2_MANIFEST_PATH, ADDED2_DIR),
    ):
        manifest_df = pd.read_csv(manifest_path)
        for _, row in manifest_df.iterrows():
            wind_speed = row.get("wind_speed")
            rpm = row.get("rpm")
            if pd.isna(wind_speed) or pd.isna(rpm):
                continue
            case_id = int(row["case_id"])
            rows.append(
                DatasetRecord(
                    case_id=case_id,
                    display_name=str(row["display_name"]),
                    file_name=f"工况{case_id}.csv",
                    file_path=data_dir / f"工况{case_id}.csv",
                    wind_speed=float(wind_speed),
                    rpm=float(rpm),
                    is_labeled=True,
                    original_file_name=str(row.get("original_file_name", "")),
                    label_source=str(row.get("label_source", "")),
                    notes=str(row.get("notes", "")),
                )
            )
    return sorted(rows, key=lambda record: record.case_id)


def build_vib_ft_feature_frame(
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    config: WindowConfig,
) -> pd.DataFrame:
    frames = [
        build_case_feature_frame(record, cleaned_signal_frames[record.case_id], config)
        for record in records
    ]
    return pd.concat(frames, ignore_index=True)


def run_tabular_protocols(feature_df: pd.DataFrame, protocol_name: str, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labeled_df = feature_df.loc[feature_df["wind_speed"].notna()].copy().reset_index(drop=True)
    case_ids = labeled_df["case_id"].drop_duplicates().tolist()

    for holdout_case_id in case_ids:
        train_df = labeled_df.loc[labeled_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_df = labeled_df.loc[labeled_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        protocol = "external_loocv"
        rows.extend(build_tabular_prediction_rows(protocol, protocol_name, train_df, test_df, feature_columns))

    train_df = labeled_df.loc[labeled_df["case_id"].between(21, 24)].copy().reset_index(drop=True)
    test_df = labeled_df.loc[labeled_df["case_id"].between(25, 30)].copy().reset_index(drop=True)
    rows.extend(build_tabular_prediction_rows("added_to_added2", protocol_name, train_df, test_df, feature_columns))
    return pd.DataFrame(rows)


def build_tabular_prediction_rows(
    protocol: str,
    variant_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(
        train_df[feature_columns].to_numpy(dtype=float),
        train_df["wind_speed"].to_numpy(dtype=float),
    )
    window_pred = estimator.predict(test_df[feature_columns].to_numpy(dtype=float))

    pred_df = test_df[
        ["case_id", "file_name", "wind_speed", "rpm"]
    ].copy()
    pred_df["pred_wind_speed_window"] = window_pred
    case_pred_df = (
        pred_df.groupby(["case_id", "file_name", "wind_speed", "rpm"], as_index=False)["pred_wind_speed_window"]
        .mean()
        .rename(columns={"wind_speed": "true_wind_speed", "pred_wind_speed_window": "pred_wind_speed"})
    )
    case_pred_df["domain"] = np.where(case_pred_df["case_id"] <= 24, "added", "added2")
    case_pred_df["protocol"] = protocol
    case_pred_df["variant_name"] = variant_name
    case_pred_df["signed_error"] = case_pred_df["pred_wind_speed"] - case_pred_df["true_wind_speed"]
    case_pred_df["abs_error"] = case_pred_df["signed_error"].abs()
    return case_pred_df.to_dict(orient="records")


def run_rpm_knn4_protocols(records: list[DatasetRecord]) -> pd.DataFrame:
    case_df = pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "domain": "added" if record.case_id <= 24 else "added2",
                "true_wind_speed": float(record.wind_speed),
                "rpm": float(record.rpm),
            }
            for record in records
        ]
    )
    rows: list[dict[str, object]] = []
    for holdout_case_id in case_df["case_id"].tolist():
        train_df = case_df.loc[case_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_row = case_df.loc[case_df["case_id"] == holdout_case_id].iloc[0]
        rows.append(build_rpm_knn_row("external_loocv", train_df, test_row))

    train_df = case_df.loc[case_df["case_id"].between(21, 24)].copy().reset_index(drop=True)
    for _, test_row in case_df.loc[case_df["case_id"].between(25, 30)].iterrows():
        rows.append(build_rpm_knn_row("added_to_added2", train_df, test_row))
    return pd.DataFrame(rows)


def build_rpm_knn_row(protocol: str, train_df: pd.DataFrame, test_row: pd.Series) -> dict[str, object]:
    pred = predict_rpm_knn4(train_df, float(test_row["rpm"]))
    signed_error = pred - float(test_row["true_wind_speed"])
    return {
        "protocol": protocol,
        "variant_name": "rpm_knn4",
        "case_id": int(test_row["case_id"]),
        "file_name": str(test_row["file_name"]),
        "domain": str(test_row["domain"]),
        "true_wind_speed": float(test_row["true_wind_speed"]),
        "rpm": float(test_row["rpm"]),
        "pred_wind_speed": float(pred),
        "signed_error": float(signed_error),
        "abs_error": float(abs(signed_error)),
    }


def predict_rpm_knn4(train_df: pd.DataFrame, rpm_value: float) -> float:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["true_wind_speed"].to_numpy(dtype=float, copy=False)
    distances = np.abs(train_rpm - rpm_value)
    order = np.argsort(distances)[: min(K_NEIGHBORS, len(train_df))]
    weights = 1.0 / np.maximum(distances[order], EPS)
    weights = weights / weights.sum()
    return float(np.dot(weights, train_wind[order]))


def load_071_reference_predictions() -> pd.DataFrame | None:
    if not REFERENCE_071_PRED_PATH.exists():
        return None
    pred_df = pd.read_csv(REFERENCE_071_PRED_PATH, encoding="utf-8-sig")
    pred_df = pred_df.loc[
        pred_df["variant_name"].isin(["rpm_knn4_plus_embedding_residual_ridge"])
    ].copy()
    if pred_df.empty:
        return None
    pred_df = pred_df.rename(columns={"domain": "domain"})
    pred_df = pred_df[
        [
            "protocol",
            "variant_name",
            "case_id",
            "file_name",
            "domain",
            "true_wind_speed",
            "rpm",
            "pred_wind_speed",
            "signed_error",
            "abs_error",
        ]
    ].copy()
    pred_df["variant_name"] = "071_reference__rpm_knn4_plus_embedding_residual_ridge"
    return pred_df


def summarize_block(block: pd.DataFrame) -> dict[str, object]:
    true_values = block["true_wind_speed"].to_numpy(dtype=float)
    pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    return {
        "case_count": int(len(block)),
        "case_mae": float(np.mean(np.abs(signed_error))),
        "case_rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
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
    return (
        pd.DataFrame(rows)
        .sort_values(["protocol", "domain", "case_mae", "case_rmse", "variant_name"])
        .reset_index(drop=True)
    )


def write_summary_markdown(output_path: Path, summary_by_protocol_df: pd.DataFrame, summary_by_protocol_and_domain_df: pd.DataFrame) -> None:
    lines = [f"# {TRY_NAME}", "", "## Summary By Protocol", ""]
    for protocol, block in summary_by_protocol_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
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
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
