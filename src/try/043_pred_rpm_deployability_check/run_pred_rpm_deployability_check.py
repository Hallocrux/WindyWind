from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY019_ROOT = REPO_ROOT / "src" / "try" / "019_tinytcn_rpm_regression"
TRY041_OUTPUT = REPO_ROOT / "outputs" / "try" / "041_rpm_vs_learned_midband_check" / "summary.csv"
TRY024_CASE_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "024_tinytcn_rpm_fine_window_scan"
    / "rpm_fine_window_scan_case_level_predictions.csv"
)
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY019_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase3_end_to_end_lib import build_raw_window_dataset
from tinytcn_rpm_lib import (  # type: ignore[import-not-found]
    TinyTCN,
    TorchTrainConfig,
    evaluate_tinytcn_rpm_loco,
    normalize_windows_by_channel,
    summarize_rpm_predictions,
    train_torch_model,
)

from src.current.data_loading import (
    DatasetRecord,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig

TRY_NAME = "043_pred_rpm_deployability_check"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
WINDOW_CONFIG_MAP = {
    "2.0s": WindowConfig(sampling_rate=50.0, window_size=100, step_size=50),
    "3.0s": WindowConfig(sampling_rate=50.0, window_size=150, step_size=75),
    "5.0s": WindowConfig(sampling_rate=50.0, window_size=250, step_size=125),
}
DEFAULT_WINDOW_LABELS = ["3.0s", "2.0s", "5.0s"]
DEFAULT_MAP_METHODS = ["rpm_knn4", "rpm_linear", "ridge_rpm_to_wind"]
LEARNED_MIDBAND_REFERENCE_VARIANT = "tinytcn_all_channels_midband_3_0_6_0hz"


@dataclass(frozen=True)
class WindMappingSpec:
    input_rpm_source: str
    window_label: str
    mapping_method: str

    @property
    def variant_name(self) -> str:
        if self.input_rpm_source == "true_rpm":
            return f"true_rpm__to__{self.mapping_method}"
        return f"pred_rpm_{self.window_label}__to__{self.mapping_method}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 pred_rpm 可部署链路。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--window-labels",
        nargs="+",
        default=DEFAULT_WINDOW_LABELS,
        help="需要评估的 rpm 预测窗长标签，默认 3.0s 2.0s 5.0s。",
    )
    parser.add_argument(
        "--map-methods",
        nargs="+",
        default=DEFAULT_MAP_METHODS,
        choices=DEFAULT_MAP_METHODS,
        help="需要评估的 rpm->wind 标量映射。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    window_labels = validate_window_labels(args.window_labels)
    torch.use_deterministic_algorithms(True)
    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    rpm_case_rows: list[pd.DataFrame] = []
    final_pred_rpm_by_window: dict[str, pd.DataFrame] = {}
    added_pred_rpm_by_window: dict[str, pd.DataFrame] = {}
    final_case_info = build_case_info_frame(final_records, domain="final_loco")
    added_case_info = build_case_info_frame(added_records, domain="added_external")
    precomputed_final_rpm = load_precomputed_final_loco_rpm(window_labels)

    rpm_case_rows.append(
        build_true_rpm_case_predictions(final_case_info, protocol="final_loco")
    )
    rpm_case_rows.append(
        build_true_rpm_case_predictions(added_case_info, protocol="added_external")
    )

    for window_index, window_label in enumerate(window_labels):
        window_config = WINDOW_CONFIG_MAP[window_label]
        final_pred_rpm_df = precomputed_final_rpm.get(window_label)
        if final_pred_rpm_df is None:
            final_pred_rpm_df = predict_final_loco_case_rpm(
                records=final_records,
                cleaned_signal_frames=cleaned_signal_frames,
                window_config=window_config,
                train_config=train_config,
                seed=args.random_seed + 97 * (window_index + 1),
            )
        final_pred_rpm_df = final_pred_rpm_df.merge(
            final_case_info[["case_id", "true_wind_speed"]],
            on="case_id",
            how="left",
        )
        final_pred_rpm_df["domain"] = "final_loco"
        final_pred_rpm_df["protocol"] = "final_loco"
        final_pred_rpm_df["window_label"] = window_label
        final_pred_rpm_df["rpm_model_name"] = f"TinyTCN@{window_label}"
        final_pred_rpm_df["input_rpm_source"] = "pred_rpm"
        final_pred_rpm_df["rpm_error"] = (
            final_pred_rpm_df["pred_rpm"] - final_pred_rpm_df["true_rpm"]
        )
        final_pred_rpm_df["rpm_abs_error"] = final_pred_rpm_df["rpm_error"].abs()
        final_pred_rpm_by_window[window_label] = final_pred_rpm_df.copy()
        rpm_case_rows.append(final_pred_rpm_df)

        added_pred_rpm_df = predict_external_case_rpm(
            train_records=final_records,
            eval_records=added_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            train_config=train_config,
            seed=args.random_seed + 197 * (window_index + 1),
        )
        added_pred_rpm_df["domain"] = "added_external"
        added_pred_rpm_df["protocol"] = "added_external"
        added_pred_rpm_df["window_label"] = window_label
        added_pred_rpm_df["rpm_model_name"] = f"TinyTCN@{window_label}"
        added_pred_rpm_df["input_rpm_source"] = "pred_rpm"
        added_pred_rpm_df["rpm_error"] = (
            added_pred_rpm_df["pred_rpm"] - added_pred_rpm_df["true_rpm"]
        )
        added_pred_rpm_df["rpm_abs_error"] = added_pred_rpm_df["rpm_error"].abs()
        added_pred_rpm_by_window[window_label] = added_pred_rpm_df.copy()
        rpm_case_rows.append(added_pred_rpm_df)

    rpm_case_level_df = pd.concat(rpm_case_rows, ignore_index=True)
    rpm_case_level_df = rpm_case_level_df[
        [
            "domain",
            "protocol",
            "case_id",
            "file_name",
            "true_wind_speed",
            "true_rpm",
            "pred_rpm",
            "rpm_error",
            "rpm_abs_error",
            "input_rpm_source",
            "window_label",
            "rpm_model_name",
        ]
    ].sort_values(["domain", "input_rpm_source", "window_label", "case_id"])

    final_wind_case_df = build_final_loco_wind_predictions(
        final_case_info=final_case_info,
        final_pred_rpm_by_window=final_pred_rpm_by_window,
        map_methods=args.map_methods,
    )
    added_wind_case_df = build_added_wind_predictions(
        train_case_info=final_case_info,
        added_case_info=added_case_info,
        added_pred_rpm_by_window=added_pred_rpm_by_window,
        map_methods=args.map_methods,
    )
    rpm_to_wind_case_level_df = pd.concat(
        [final_wind_case_df, added_wind_case_df], ignore_index=True
    )
    rpm_to_wind_case_level_df["signed_error"] = (
        rpm_to_wind_case_level_df["pred_wind_speed"]
        - rpm_to_wind_case_level_df["true_wind_speed"]
    )
    rpm_to_wind_case_level_df["abs_error"] = rpm_to_wind_case_level_df["signed_error"].abs()

    summary_df = build_wind_summary(rpm_to_wind_case_level_df)
    gap_df = build_gap_summary(summary_df, load_learned_midband_reference())
    write_summary_markdown(output_dir / "summary.md", summary_df, gap_df, window_labels)

    rpm_case_level_df.to_csv(
        output_dir / "rpm_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rpm_to_wind_case_level_df.to_csv(
        output_dir / "rpm_to_wind_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        output_dir / "rpm_to_wind_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_df.to_csv(
        output_dir / "deployable_vs_true_rpm_gap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("043 pred_rpm 可部署性验证已完成。")
    print(f"输出目录: {output_dir}")
    best_added = summary_df.loc[
        summary_df["domain"] == "added_external"
    ].sort_values(["case_mae", "variant_name"]).iloc[0]
    best_final = summary_df.loc[
        summary_df["domain"] == "final_loco"
    ].sort_values(["case_mae", "variant_name"]).iloc[0]
    print(
        f"best final: {best_final['variant_name']} | case_mae={best_final['case_mae']:.4f}"
    )
    print(
        f"best added: {best_added['variant_name']} | case_mae={best_added['case_mae']:.4f}"
    )


def validate_window_labels(window_labels: list[str]) -> list[str]:
    invalid = [label for label in window_labels if label not in WINDOW_CONFIG_MAP]
    if invalid:
        raise ValueError(f"未知窗长标签: {invalid}")
    seen: set[str] = set()
    ordered: list[str] = []
    for label in window_labels:
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def load_added_records() -> list[DatasetRecord]:
    manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=ADDED_STANDARD_DIR / f"工况{case_id}.csv",
                wind_speed=float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None,
                rpm=float(row["rpm"]) if not pd.isna(row["rpm"]) else None,
                is_labeled=not pd.isna(row["wind_speed"]) and not pd.isna(row["rpm"]),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return records


def load_precomputed_final_loco_rpm(
    window_labels: list[str],
) -> dict[str, pd.DataFrame]:
    if not TRY024_CASE_OUTPUT.exists():
        return {}
    case_df = pd.read_csv(TRY024_CASE_OUTPUT)
    result: dict[str, pd.DataFrame] = {}
    for window_label in window_labels:
        block = case_df.loc[case_df["window_label"] == window_label].copy()
        if block.empty:
            continue
        result[window_label] = block.rename(columns={"pred_mean": "pred_rpm"})[
            ["case_id", "file_name", "true_rpm", "pred_rpm"]
        ].copy()
    return result


def build_case_info_frame(records: list[DatasetRecord], domain: str) -> pd.DataFrame:
    rows = [
        {
            "domain": domain,
            "case_id": record.case_id,
            "file_name": record.file_name,
            "true_wind_speed": float(record.wind_speed),
            "true_rpm": float(record.rpm),
        }
        for record in records
    ]
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_true_rpm_case_predictions(
    case_info_df: pd.DataFrame,
    *,
    protocol: str,
) -> pd.DataFrame:
    result = case_info_df.copy()
    result["protocol"] = protocol
    result["pred_rpm"] = result["true_rpm"]
    result["rpm_error"] = 0.0
    result["rpm_abs_error"] = 0.0
    result["input_rpm_source"] = "true_rpm"
    result["window_label"] = "oracle"
    result["rpm_model_name"] = "true_rpm"
    return result


def predict_final_loco_case_rpm(
    *,
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    train_config: TorchTrainConfig,
    seed: int,
) -> pd.DataFrame:
    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, window_config)
    prediction_frame = evaluate_tinytcn_rpm_loco(
        raw_dataset,
        train_config=train_config,
        random_seed=seed,
    )
    _, case_df = summarize_rpm_predictions(prediction_frame, "TinyTCN")
    return case_df.rename(columns={"pred_mean": "pred_rpm"}).copy()


def predict_external_case_rpm(
    *,
    train_records: list[DatasetRecord],
    eval_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    train_config: TorchTrainConfig,
    seed: int,
) -> pd.DataFrame:
    train_dataset = build_raw_window_dataset(train_records, cleaned_signal_frames, window_config)
    eval_dataset = build_raw_window_dataset(eval_records, cleaned_signal_frames, window_config)
    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["rpm"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows
    X_train_norm, X_eval_norm = normalize_windows_by_channel(X_train, X_eval)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = TinyTCN(in_channels=X_train.shape[1]).to(device)
    train_torch_model(
        model=model,
        X_train=X_train_norm,
        y_train=y_train,
        X_valid=X_train_norm,
        y_valid=y_train,
        config=train_config,
        device=device,
    )
    with torch.no_grad():
        pred = model(torch.from_numpy(X_eval_norm).to(device)).cpu().numpy()

    pred_df = eval_dataset.meta_df[
        ["case_id", "file_name", "wind_speed", "rpm"]
    ].copy()
    pred_df["pred_rpm"] = pred
    case_df = (
        pred_df.groupby(["case_id", "file_name", "wind_speed", "rpm"], as_index=False)[
            "pred_rpm"
        ]
        .mean()
        .rename(
            columns={
                "wind_speed": "true_wind_speed",
                "rpm": "true_rpm",
            }
        )
    )
    return case_df


def build_final_loco_wind_predictions(
    *,
    final_case_info: pd.DataFrame,
    final_pred_rpm_by_window: dict[str, pd.DataFrame],
    map_methods: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    case_lookup_by_window = {
        window_label: frame.set_index("case_id")
        for window_label, frame in final_pred_rpm_by_window.items()
    }
    for case_id, test_row in final_case_info.set_index("case_id").iterrows():
        train_df = final_case_info.loc[final_case_info["case_id"] != case_id].copy()
        for mapping_method in map_methods:
            specs: list[WindMappingSpec] = [WindMappingSpec("true_rpm", "oracle", mapping_method)]
            specs.extend(
                WindMappingSpec("pred_rpm", window_label, mapping_method)
                for window_label in final_pred_rpm_by_window
                if should_include_variant(window_label=window_label, mapping_method=mapping_method)
            )
            for spec in specs:
                input_rpm = float(test_row["true_rpm"])
                if spec.input_rpm_source == "pred_rpm":
                    input_rpm = float(case_lookup_by_window[spec.window_label].loc[case_id, "pred_rpm"])
                pred_wind = predict_wind_from_rpm(
                    train_rpm=train_df["true_rpm"].to_numpy(dtype=float),
                    train_wind=train_df["true_wind_speed"].to_numpy(dtype=float),
                    input_rpm=np.array([input_rpm], dtype=float),
                    mapping_method=mapping_method,
                )[0]
                rows.append(
                    {
                        "domain": "final_loco",
                        "protocol": "final_loco",
                        "case_id": int(case_id),
                        "file_name": str(test_row["file_name"]),
                        "true_wind_speed": float(test_row["true_wind_speed"]),
                        "true_rpm": float(test_row["true_rpm"]),
                        "input_rpm_value": float(input_rpm),
                        "input_rpm_source": spec.input_rpm_source,
                        "window_label": spec.window_label,
                        "mapping_method": mapping_method,
                        "variant_name": spec.variant_name,
                        "pred_wind_speed": float(pred_wind),
                    }
                )
    return pd.DataFrame(rows).sort_values(["variant_name", "case_id"]).reset_index(drop=True)


def build_added_wind_predictions(
    *,
    train_case_info: pd.DataFrame,
    added_case_info: pd.DataFrame,
    added_pred_rpm_by_window: dict[str, pd.DataFrame],
    map_methods: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_rpm = train_case_info["true_rpm"].to_numpy(dtype=float)
    train_wind = train_case_info["true_wind_speed"].to_numpy(dtype=float)
    added_lookup_by_window = {
        window_label: frame.set_index("case_id")
        for window_label, frame in added_pred_rpm_by_window.items()
    }
    for _, test_row in added_case_info.iterrows():
        case_id = int(test_row["case_id"])
        for mapping_method in map_methods:
            specs: list[WindMappingSpec] = [WindMappingSpec("true_rpm", "oracle", mapping_method)]
            specs.extend(
                WindMappingSpec("pred_rpm", window_label, mapping_method)
                for window_label in added_pred_rpm_by_window
                if should_include_variant(window_label=window_label, mapping_method=mapping_method)
            )
            for spec in specs:
                input_rpm = float(test_row["true_rpm"])
                if spec.input_rpm_source == "pred_rpm":
                    input_rpm = float(added_lookup_by_window[spec.window_label].loc[case_id, "pred_rpm"])
                pred_wind = predict_wind_from_rpm(
                    train_rpm=train_rpm,
                    train_wind=train_wind,
                    input_rpm=np.array([input_rpm], dtype=float),
                    mapping_method=mapping_method,
                )[0]
                rows.append(
                    {
                        "domain": "added_external",
                        "protocol": "added_external",
                        "case_id": case_id,
                        "file_name": str(test_row["file_name"]),
                        "true_wind_speed": float(test_row["true_wind_speed"]),
                        "true_rpm": float(test_row["true_rpm"]),
                        "input_rpm_value": float(input_rpm),
                        "input_rpm_source": spec.input_rpm_source,
                        "window_label": spec.window_label,
                        "mapping_method": mapping_method,
                        "variant_name": spec.variant_name,
                        "pred_wind_speed": float(pred_wind),
                    }
                )
    return pd.DataFrame(rows).sort_values(["variant_name", "case_id"]).reset_index(drop=True)


def should_include_variant(*, window_label: str, mapping_method: str) -> bool:
    if window_label == "3.0s":
        return True
    return mapping_method == "rpm_linear"


def predict_wind_from_rpm(
    *,
    train_rpm: np.ndarray,
    train_wind: np.ndarray,
    input_rpm: np.ndarray,
    mapping_method: str,
) -> np.ndarray:
    if mapping_method == "rpm_knn4":
        return np.array(
            [
                weighted_rpm_neighbor_prediction(
                    train_rpm=train_rpm,
                    train_wind=train_wind,
                    rpm_value=float(rpm_value),
                )
                for rpm_value in input_rpm
            ],
            dtype=float,
        )

    X_train = train_rpm.reshape(-1, 1)
    X_input = input_rpm.reshape(-1, 1)
    if mapping_method == "rpm_linear":
        estimator = LinearRegression()
    elif mapping_method == "ridge_rpm_to_wind":
        estimator = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
    else:
        raise ValueError(f"未知 rpm->wind 映射: {mapping_method}")
    estimator.fit(X_train, train_wind)
    return estimator.predict(X_input)


def weighted_rpm_neighbor_prediction(
    *,
    train_rpm: np.ndarray,
    train_wind: np.ndarray,
    rpm_value: float,
) -> float:
    distances = np.abs(train_rpm - rpm_value)
    order = np.argsort(distances)[:4]
    selected_distances = distances[order]
    selected_wind = train_wind[order]
    weights = 1.0 / np.maximum(selected_distances, 1.0)
    return float(np.average(selected_wind, weights=weights))


def build_wind_summary(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, variant_name), block in case_level_df.groupby(["domain", "variant_name"], sort=False):
        rows.append(
            {
                "domain": domain,
                "protocol": block["protocol"].iloc[0],
                "variant_name": variant_name,
                "input_rpm_source": block["input_rpm_source"].iloc[0],
                "window_label": block["window_label"].iloc[0],
                "mapping_method": block["mapping_method"].iloc[0],
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case_count": int(len(block)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["domain", "case_mae", "case_rmse", "variant_name"])
        .reset_index(drop=True)
    )


def build_gap_summary(
    summary_df: pd.DataFrame,
    learned_midband_added_case_mae_ref: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, domain_df in summary_df.groupby("domain", sort=False):
        upper_lookup = domain_df.loc[
            domain_df["input_rpm_source"] == "true_rpm",
            ["mapping_method", "case_mae", "case_rmse"],
        ].rename(
            columns={
                "case_mae": "upper_case_mae",
                "case_rmse": "upper_case_rmse",
            }
        )
        pred_df = domain_df.loc[domain_df["input_rpm_source"] == "pred_rpm"].copy()
        merged = pred_df.merge(upper_lookup, on="mapping_method", how="left")
        for _, row in merged.iterrows():
            output_row = {
                "domain": domain,
                "variant_name": row["variant_name"],
                "window_label": row["window_label"],
                "mapping_method": row["mapping_method"],
                "deployable_case_mae": float(row["case_mae"]),
                "upper_case_mae": float(row["upper_case_mae"]),
                "case_mae_gap_vs_true_rpm": float(row["case_mae"] - row["upper_case_mae"]),
                "deployable_case_rmse": float(row["case_rmse"]),
                "upper_case_rmse": float(row["upper_case_rmse"]),
                "case_rmse_gap_vs_true_rpm": float(row["case_rmse"] - row["upper_case_rmse"]),
                "learned_midband_added_case_mae_ref": np.nan,
                "delta_vs_learned_midband_added_case_mae_ref": np.nan,
            }
            if domain == "added_external" and learned_midband_added_case_mae_ref is not None:
                output_row["learned_midband_added_case_mae_ref"] = float(
                    learned_midband_added_case_mae_ref
                )
                output_row["delta_vs_learned_midband_added_case_mae_ref"] = float(
                    row["case_mae"] - learned_midband_added_case_mae_ref
                )
            rows.append(output_row)
    return (
        pd.DataFrame(rows)
        .sort_values(["domain", "deployable_case_mae", "variant_name"])
        .reset_index(drop=True)
    )


def load_learned_midband_reference() -> float | None:
    if not TRY041_OUTPUT.exists():
        return None
    summary_df = pd.read_csv(TRY041_OUTPUT)
    match_df = summary_df.loc[
        summary_df["variant_name"] == LEARNED_MIDBAND_REFERENCE_VARIANT
    ]
    if match_df.empty:
        return None
    return float(match_df.iloc[0]["case_mae"])


def write_summary_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    window_labels: list[str],
) -> None:
    final_best = summary_df.loc[summary_df["domain"] == "final_loco"].iloc[0]
    added_best = summary_df.loc[summary_df["domain"] == "added_external"].iloc[0]
    final_deployable_best = summary_df.loc[
        (summary_df["domain"] == "final_loco")
        & (summary_df["input_rpm_source"] == "pred_rpm")
    ].iloc[0]
    added_deployable_best = summary_df.loc[
        (summary_df["domain"] == "added_external")
        & (summary_df["input_rpm_source"] == "pred_rpm")
    ].iloc[0]
    primary_3s_gap = gap_df.loc[
        (gap_df["domain"] == "added_external")
        & (gap_df["window_label"] == "3.0s")
    ].sort_values(["deployable_case_mae", "variant_name"]).reset_index(drop=True)

    lines = [
        "# 043 `pred_rpm` 可部署性验证",
        "",
        f"- 评估窗长：`{', '.join(window_labels)}`",
        f"- `final` 全域最优：`{final_best['variant_name']}` | case_mae=`{final_best['case_mae']:.4f}`",
        f"- `added` 全域最优：`{added_best['variant_name']}` | case_mae=`{added_best['case_mae']:.4f}`",
        f"- `final` 最优可部署链：`{final_deployable_best['variant_name']}` | case_mae=`{final_deployable_best['case_mae']:.4f}`",
        f"- `added` 最优可部署链：`{added_deployable_best['variant_name']}` | case_mae=`{added_deployable_best['case_mae']:.4f}`",
        "",
        "## `added` 主候选对照",
        "",
    ]
    for _, row in primary_3s_gap.iterrows():
        extra = ""
        if not pd.isna(row["delta_vs_learned_midband_added_case_mae_ref"]):
            extra = (
                ", "
                f"vs learned_midband_ref=`{row['delta_vs_learned_midband_added_case_mae_ref']:+.4f}`"
            )
        lines.append(
            f"- `{row['variant_name']}`: deployable case_mae=`{row['deployable_case_mae']:.4f}`, upper=`{row['upper_case_mae']:.4f}`, gap=`{row['case_mae_gap_vs_true_rpm']:+.4f}`{extra}"
        )

    lines.extend(["", "## 当前判断", ""])
    if added_deployable_best["case_mae"] <= added_best["case_mae"] + 1e-12:
        lines.append("- 当前最优的 added 方案已经是可部署链路，不再需要依赖真实 rpm。")
    else:
        lines.append("- 当前 added 的最优结果仍来自真实 rpm 上界；可部署链路存在可量化退化。")
    if final_deployable_best["case_mae"] <= final_best["case_mae"] + 1e-12:
        lines.append("- 当前 final 的最优结果也可以由可部署链路达到。")
    else:
        lines.append("- 当前 final 上界与可部署链路之间仍存在差距，后续应重点关注 pred_rpm 偏差。")
    lines.append(
        "- 本轮 `ridge_rpm_to_wind` 采用二次平滑映射，用来测试是否比邻域型 `knn4` 更能容忍 `pred_rpm` 噪声。"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
