from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRY009_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY009_ROOT))
if str(TRY012_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY012_ROOT))
if str(TRY013_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY013_ROOT))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase3_cnn_tcn_lib import (
    TorchTrainConfig,
    build_raw_window_dataset,
    evaluate_torch_model_loco,
    predict_torch_model_unlabeled,
    summarize_predictions,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "Baseline_TinyTCN"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())
    train_config = TorchTrainConfig()

    prediction_frame = evaluate_torch_model_loco(raw_dataset, "TinyTCN", train_config)
    summary_row, case_df = summarize_predictions(prediction_frame, "TinyTCN")
    summary_df = pd.DataFrame([summary_row])
    unlabeled_df = predict_torch_model_unlabeled(raw_dataset, "TinyTCN", train_config)

    summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(OUTPUT_DIR / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    unlabeled_df.to_csv(OUTPUT_DIR / "unlabeled_predictions.csv", index=False, encoding="utf-8-sig")

    print("Baseline TinyTCN 已完成。")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"case_mae: {summary_row['case_mae']:.4f}")
    if not unlabeled_df.empty:
        first_row = unlabeled_df.iloc[0]
        print(f"{first_row['file_name']} 预测风速: {first_row['predicted_wind_speed']:.4f} m/s")
