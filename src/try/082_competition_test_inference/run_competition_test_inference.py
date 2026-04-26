from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "082_competition_test_inference"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
TEST_DATA_PATH = REPO_ROOT / "data" / "test" / "竞赛预测风速工况.csv"
EXTERNAL_FEATURE_TABLE_PATH = (
    REPO_ROOT / "outputs" / "try" / "079_repo_fft_sideinfo_in_071_residual" / "external_feature_table.csv"
)
FFT_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "043_1_fft_rpm_algorithm_search" / "run_fft_rpm_algorithm_search.py"
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY066_ROOT = REPO_ROOT / "src" / "try" / "066_reuse_embedding_domain_split"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
K_NEIGHBORS = 4
EPS = 1e-6
TEST_CASE_ID = 999
TEST_FILE_NAME = "竞赛预测风速工况.csv"
FFT_SOURCE_COLUMN_ORDER = ["fft_peak_1x_whole", "window_peak_1x_conf_8s"]

for path in (REPO_ROOT, TRY009_ROOT, TRY066_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase1_feature_groups_lib import build_feature_frame, get_group_feature_columns  # noqa: E402
from reuse_embedding_domain_common import (  # noqa: E402
    build_embedding_case_table,
    load_fixed_window_embeddings,
    load_source_catalog,
    load_try053_module,
)
from src.current.data_loading import (  # noqa: E402
    DatasetRecord,
    get_common_signal_columns,
    load_clean_signal_frame,
    prepare_clean_signal_frame,
)
from src.current.features import (  # noqa: E402
    WindowConfig,
    build_case_feature_frame,
    get_vibration_feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对竞赛测试工况运行 071 与若干候选模型。")
    parser.add_argument("--test-path", type=Path, default=TEST_DATA_PATH)
    parser.add_argument("--rpm", type=float, required=True, help="竞赛测试工况给定转速。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_external_df = load_external_feature_table()
    train_records = build_external_records()
    test_record = build_test_record(args.test_path, args.rpm)

    catalog = load_source_catalog()
    embedding_common_signal_columns = get_common_signal_columns(catalog.all_records)
    validate_test_columns(test_record, embedding_common_signal_columns)

    train_cleaned_frames = {
        record.case_id: load_clean_signal_frame(record, embedding_common_signal_columns)
        for record in train_records
    }
    test_cleaned_frame, test_cleaning_stats = prepare_clean_signal_frame(test_record, embedding_common_signal_columns)

    test_embedding_df = build_test_embedding_feature_row(
        test_record=test_record,
        test_cleaned_frame=test_cleaned_frame,
    )
    test_fft_sideinfo = estimate_test_repo_fft_sideinfo(
        test_record=test_record,
        test_cleaned_frame=test_cleaned_frame,
        true_rpm=args.rpm,
    )
    test_feature_row = attach_fft_sideinfo_to_test_embedding(test_embedding_df, test_fft_sideinfo)

    predictions = []
    predictions.extend(run_embedding_candidates(train_external_df, test_feature_row))
    predictions.extend(run_tabular_candidates(train_records, train_cleaned_frames, test_record, test_cleaned_frame))

    prediction_df = pd.DataFrame(predictions).sort_values(["sort_order", "variant_name"]).reset_index(drop=True)
    prediction_df["rank_by_pred"] = np.arange(1, len(prediction_df) + 1)

    test_feature_row.to_csv(output_dir / "test_feature_row.csv", index=False, encoding="utf-8-sig")
    prediction_df.to_csv(output_dir / "prediction_summary.csv", index=False, encoding="utf-8-sig")
    write_signal_inventory(
        output_path=output_dir / "test_signal_inventory.json",
        test_record=test_record,
        test_cleaned_frame=test_cleaned_frame,
        test_cleaning_stats=test_cleaning_stats,
        test_fft_sideinfo=test_fft_sideinfo,
        train_external_df=train_external_df,
    )
    write_summary_markdown(
        output_path=output_dir / "prediction_summary.md",
        prediction_df=prediction_df,
        test_record=test_record,
        test_fft_sideinfo=test_fft_sideinfo,
        train_external_df=train_external_df,
    )

    print("082 competition test inference 已完成。")
    print(f"输出目录: {output_dir}")
    for _, row in prediction_df.iterrows():
        print(f"{row['variant_name']}: pred_wind_speed={row['pred_wind_speed']:.4f}")


def load_external_feature_table() -> pd.DataFrame:
    if not EXTERNAL_FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(f"缺少 external feature table: {EXTERNAL_FEATURE_TABLE_PATH}")
    df = pd.read_csv(EXTERNAL_FEATURE_TABLE_PATH, encoding="utf-8-sig")
    df = (
        df.loc[df["raw_source_domain"].isin(["added", "added2"]) & df["is_labeled"]]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("external feature table 中没有 added/added2 带标签样本。")
    return df


def build_external_records() -> list[DatasetRecord]:
    rows: list[DatasetRecord] = []
    for manifest_path, data_dir in (
        (REPO_ROOT / "data" / "added" / "dataset_manifest.csv", REPO_ROOT / "data" / "added" / "standardized_datasets"),
        (REPO_ROOT / "data" / "added2" / "dataset_manifest.csv", REPO_ROOT / "data" / "added2" / "standardized_datasets"),
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


def build_test_record(test_path: Path, rpm: float) -> DatasetRecord:
    if not test_path.exists():
        raise FileNotFoundError(f"未找到测试文件: {test_path}")
    return DatasetRecord(
        case_id=TEST_CASE_ID,
        display_name="竞赛测试工况",
        file_name=TEST_FILE_NAME,
        file_path=test_path,
        wind_speed=None,
        rpm=float(rpm),
        is_labeled=False,
        original_file_name=test_path.name,
        label_source="user_provided_rpm_2026-04-09",
        notes="competition_test_inference",
    )


def validate_test_columns(test_record: DatasetRecord, required_signal_columns: list[str]) -> None:
    test_columns = pd.read_csv(test_record.file_path, nrows=0).columns.tolist()
    missing_columns = [column for column in required_signal_columns if column not in test_columns]
    if missing_columns:
        raise ValueError(f"测试文件缺少必要列: {missing_columns}")


def build_test_embedding_feature_row(
    *,
    test_record: DatasetRecord,
    test_cleaned_frame: pd.DataFrame,
) -> pd.DataFrame:
    try053 = load_try053_module()
    fixed_window_embeddings = load_fixed_window_embeddings(
        try053=try053,
        export_records=[test_record],
        cleaned_signal_frames={test_record.case_id: test_cleaned_frame},
    )
    record_df = pd.DataFrame(
        [
            {
                "case_id": int(test_record.case_id),
                "file_name": str(test_record.file_name),
                "display_name": str(test_record.display_name),
                "raw_source_domain": "competition_test",
                "wind_speed": np.nan,
                "rpm": float(test_record.rpm),
                "is_labeled": False,
                "notes": str(test_record.notes),
            }
        ]
    )
    return build_embedding_case_table(record_df, fixed_window_embeddings)


def load_fft_module():
    spec = importlib.util.spec_from_file_location("fft_module_082", FFT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 FFT 脚本: {FFT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def estimate_test_repo_fft_sideinfo(
    *,
    test_record: DatasetRecord,
    test_cleaned_frame: pd.DataFrame,
    true_rpm: float,
) -> dict[str, object]:
    fft_module = load_fft_module()
    spec_map = {spec.variant_name: spec for spec in fft_module.build_variant_specs()}
    whole_est = fft_module.estimate_record_rpm(
        record=test_record,
        frame=test_cleaned_frame,
        spec=spec_map["fft_peak_1x_whole"],
    )
    window_est = fft_module.estimate_record_rpm(
        record=test_record,
        frame=test_cleaned_frame,
        spec=spec_map["window_peak_1x_conf_8s"],
    )

    pred_whole = float(whole_est.pred_rpm)
    pred_window = float(window_est.pred_rpm)
    if np.isclose(pred_whole, pred_window, atol=1e-9):
        use_window = False
    else:
        use_window = max(pred_whole, pred_window) < 150.0 or pred_window > pred_whole
    hybrid_rpm = pred_window if use_window else pred_whole
    hybrid_source = "window_peak_1x_conf_8s" if use_window else "fft_peak_1x_whole"
    hybrid_confidence = float(window_est.confidence if use_window else whole_est.confidence)

    return {
        "repo_fft_rpm": float(hybrid_rpm),
        "repo_fft_source": hybrid_source,
        "repo_fft_confidence": hybrid_confidence,
        "repo_fft_whole_rpm": pred_whole,
        "repo_fft_window8_rpm": pred_window,
        "repo_delta_rpm": float(hybrid_rpm - true_rpm),
        "repo_abs_delta_rpm": abs(float(hybrid_rpm - true_rpm)),
    }


def attach_fft_sideinfo_to_test_embedding(
    test_embedding_df: pd.DataFrame,
    test_fft_sideinfo: dict[str, object],
) -> pd.DataFrame:
    df = test_embedding_df.copy()
    for key, value in test_fft_sideinfo.items():
        df[key] = value
    for source_name in FFT_SOURCE_COLUMN_ORDER:
        df[f"repo_fft_source__{source_name}"] = float(test_fft_sideinfo["repo_fft_source"] == source_name)
    return df


def run_embedding_candidates(train_external_df: pd.DataFrame, test_feature_row: pd.DataFrame) -> list[dict[str, object]]:
    embedding_columns = [column for column in train_external_df.columns if column.startswith("embedding_")]
    residual_targets = compute_internal_oof_residual_targets(train_external_df)
    base_pred = float(compute_rpm_knn_predictions(train_external_df, np.asarray([float(test_feature_row["rpm"].iloc[0])]))[0])

    pred_map = {
        "071__rpm_knn4_plus_embedding_residual_ridge": base_pred
        + float(
            fit_predict_ridge(
                train_X=train_external_df[embedding_columns].to_numpy(dtype=float),
                train_y=residual_targets,
                test_X=test_feature_row[embedding_columns].to_numpy(dtype=float),
            )[0]
        ),
        "071__embedding_ridge": float(
            fit_predict_ridge(
                train_X=train_external_df[embedding_columns].to_numpy(dtype=float),
                train_y=train_external_df["wind_speed"].to_numpy(dtype=float),
                test_X=test_feature_row[embedding_columns].to_numpy(dtype=float),
            )[0]
        ),
        "071__embedding_knn4": float(
            fit_predict_embedding_knn(
                train_X=train_external_df[embedding_columns].to_numpy(dtype=float),
                train_y=train_external_df["wind_speed"].to_numpy(dtype=float),
                test_X=test_feature_row[embedding_columns].to_numpy(dtype=float),
            )[0]
        ),
        "rpm_knn4": base_pred,
    }

    full_sideinfo_columns = [
        *embedding_columns,
        "repo_fft_rpm",
        "repo_delta_rpm",
        "repo_abs_delta_rpm",
        "repo_fft_confidence",
        "repo_fft_source__fft_peak_1x_whole",
        "repo_fft_source__window_peak_1x_conf_8s",
    ]
    pred_map["079__rpm_knn4_plus_embedding_repo_fft_sideinfo_residual_ridge"] = base_pred + float(
        fit_predict_ridge(
            train_X=train_external_df[full_sideinfo_columns].to_numpy(dtype=float),
            train_y=residual_targets,
            test_X=test_feature_row[full_sideinfo_columns].to_numpy(dtype=float),
        )[0]
    )

    rows = []
    for variant_name, pred in pred_map.items():
        rows.append(
            build_prediction_row(
                variant_name=variant_name,
                pred_wind_speed=float(pred),
                family=prediction_family(variant_name),
                train_pool="added+added2 labeled cases 21-30",
                test_feature_row=test_feature_row,
            )
        )
    return rows


def run_tabular_candidates(
    train_records: list[DatasetRecord],
    train_cleaned_frames: dict[int, pd.DataFrame],
    test_record: DatasetRecord,
    test_cleaned_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    window_config = WindowConfig()
    g6_feature_df = build_feature_frame(train_records, train_cleaned_frames, window_config)
    test_g6_feature_df = build_feature_frame([test_record], {test_record.case_id: test_cleaned_frame}, window_config)

    vib_ft_feature_df = pd.concat(
        [build_case_feature_frame(record, train_cleaned_frames[record.case_id], window_config) for record in train_records],
        ignore_index=True,
    )
    test_vib_ft_feature_df = build_case_feature_frame(test_record, test_cleaned_frame, window_config)

    g6_columns = get_group_feature_columns(g6_feature_df, "G6_TIME_FREQ_CROSS")
    vib_ft_columns = [*get_vibration_feature_columns(vib_ft_feature_df), "rpm"]

    g6_pred = fit_predict_window_ridge(
        train_df=g6_feature_df,
        test_df=test_g6_feature_df,
        feature_columns=g6_columns,
    )
    vib_ft_pred = fit_predict_window_ridge(
        train_df=vib_ft_feature_df,
        test_df=test_vib_ft_feature_df,
        feature_columns=vib_ft_columns,
    )

    test_meta_row = test_vib_ft_feature_df.iloc[[0]].copy()
    rows = [
        build_prediction_row(
            variant_name="ridge_vib_ft_rpm",
            pred_wind_speed=float(vib_ft_pred),
            family="tabular_linear",
            train_pool="added+added2 labeled cases 21-30",
            test_feature_row=test_meta_row,
        ),
        build_prediction_row(
            variant_name="tabular_reference_g6_ridge",
            pred_wind_speed=float(g6_pred),
            family="tabular_linear",
            train_pool="added+added2 labeled cases 21-30",
            test_feature_row=test_meta_row,
        ),
    ]
    return rows


def fit_predict_window_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> float:
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(
        train_df[feature_columns].to_numpy(dtype=float),
        train_df["wind_speed"].to_numpy(dtype=float),
    )
    pred = estimator.predict(test_df[feature_columns].to_numpy(dtype=float))
    return float(np.mean(pred))


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[float] = []
    for rpm_value in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm_value)
        order = np.argsort(distances)[: min(K_NEIGHBORS, len(train_df))]
        weights = 1.0 / np.maximum(distances[order], EPS)
        weights = weights / weights.sum()
        predictions.append(float(np.dot(weights, train_wind[order])))
    return np.asarray(predictions, dtype=float)


def compute_internal_oof_residual_targets(train_df: pd.DataFrame) -> np.ndarray:
    if len(train_df) <= 1:
        return np.zeros(len(train_df), dtype=float)
    residuals = []
    rpm_values = train_df["rpm"].to_numpy(dtype=float, copy=False)
    true_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    for row_idx in range(len(train_df)):
        inner_train = train_df.drop(index=train_df.index[row_idx]).reset_index(drop=True)
        pred = compute_rpm_knn_predictions(inner_train, np.asarray([rpm_values[row_idx]], dtype=float))[0]
        residuals.append(float(true_wind[row_idx] - pred))
    return np.asarray(residuals, dtype=float)


def fit_predict_ridge(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    model.fit(np.asarray(train_X, dtype=float), np.asarray(train_y, dtype=float))
    return model.predict(np.asarray(test_X, dtype=float))


def fit_predict_embedding_knn(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=min(K_NEIGHBORS, len(train_X)), weights="distance")),
        ]
    )
    model.fit(np.asarray(train_X, dtype=float), np.asarray(train_y, dtype=float))
    return model.predict(np.asarray(test_X, dtype=float))


def prediction_family(variant_name: str) -> str:
    if variant_name.startswith("071__"):
        return "embedding_residual"
    if variant_name.startswith("079__"):
        return "embedding_residual_with_fft_sideinfo"
    if variant_name == "rpm_knn4":
        return "analytic_rpm"
    return "other"


def build_prediction_row(
    *,
    variant_name: str,
    pred_wind_speed: float,
    family: str,
    train_pool: str,
    test_feature_row: pd.DataFrame,
) -> dict[str, object]:
    rpm = float(test_feature_row["rpm"].iloc[0])
    sort_order_map = {
        "071__rpm_knn4_plus_embedding_residual_ridge": 1,
        "079__rpm_knn4_plus_embedding_repo_fft_sideinfo_residual_ridge": 2,
        "rpm_knn4": 3,
        "071__embedding_ridge": 4,
        "071__embedding_knn4": 5,
        "ridge_vib_ft_rpm": 6,
        "tabular_reference_g6_ridge": 7,
    }
    return {
        "variant_name": variant_name,
        "family": family,
        "train_pool": train_pool,
        "pred_wind_speed": float(pred_wind_speed),
        "rpm": rpm,
        "sort_order": int(sort_order_map.get(variant_name, 99)),
    }


def write_signal_inventory(
    *,
    output_path: Path,
    test_record: DatasetRecord,
    test_cleaned_frame: pd.DataFrame,
    test_cleaning_stats,
    test_fft_sideinfo: dict[str, object],
    train_external_df: pd.DataFrame,
) -> None:
    payload = {
        "test_file": str(test_record.file_path),
        "test_rpm": float(test_record.rpm),
        "raw_row_count": int(len(pd.read_csv(test_record.file_path))),
        "cleaned_row_count": int(len(test_cleaned_frame)),
        "segment_count": int(test_cleaned_frame["__segment_id"].nunique()),
        "train_case_count": int(len(train_external_df)),
        "train_case_ids": train_external_df["case_id"].astype(int).tolist(),
        "cleaning_stats": {
            "leading_missing_len": int(test_cleaning_stats.leading_missing_len),
            "trailing_missing_len": int(test_cleaning_stats.trailing_missing_len),
            "edge_removed_rows": int(test_cleaning_stats.edge_removed_rows),
            "rows_after_edge_drop": int(test_cleaning_stats.rows_after_edge_drop),
            "internal_short_gap_rows": int(test_cleaning_stats.internal_short_gap_rows),
            "internal_long_gap_rows_dropped": int(test_cleaning_stats.internal_long_gap_rows_dropped),
            "continuous_segment_count": int(test_cleaning_stats.continuous_segment_count),
            "rows_after_long_gap_drop": int(test_cleaning_stats.rows_after_long_gap_drop),
        },
        "repo_fft_sideinfo": {
            key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in test_fft_sideinfo.items()
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_markdown(
    *,
    output_path: Path,
    prediction_df: pd.DataFrame,
    test_record: DatasetRecord,
    test_fft_sideinfo: dict[str, object],
    train_external_df: pd.DataFrame,
) -> None:
    lines = [
        "# competition test inference",
        "",
        "## 输入口径",
        "",
        f"- 测试文件：`{test_record.file_path}`",
        f"- 给定转速：`{float(test_record.rpm):.4f}`",
        f"- 训练池：`added + added2` 带标签工况 `21-30`，共 `10` 个 case",
        "",
        "## 仓库 FFT side-info",
        "",
        f"- `repo_fft_rpm`：`{float(test_fft_sideinfo['repo_fft_rpm']):.4f}`",
        f"- `repo_fft_whole_rpm`：`{float(test_fft_sideinfo['repo_fft_whole_rpm']):.4f}`",
        f"- `repo_fft_window8_rpm`：`{float(test_fft_sideinfo['repo_fft_window8_rpm']):.4f}`",
        f"- `repo_delta_rpm`：`{float(test_fft_sideinfo['repo_delta_rpm']):+.4f}`",
        f"- `repo_fft_source`：`{test_fft_sideinfo['repo_fft_source']}`",
        f"- `repo_fft_confidence`：`{float(test_fft_sideinfo['repo_fft_confidence']):.4f}`",
        "",
        "## 预测结果",
        "",
    ]
    for _, row in prediction_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: pred_wind_speed=`{float(row['pred_wind_speed']):.4f}`"
        )
    lines.extend(
        [
            "",
            "## 备注",
            "",
            "- `071` 是 `2026-04-09` added-first 收尾后固定的默认最佳模型；",
            "- `079` 只作为 FFT side-info 注入版候选对照；",
            "- `ridge_vib_ft_rpm` 与 `G6` 是已补测过的线性表格参考，不代表当前默认主线。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
