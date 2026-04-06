from __future__ import annotations

import pandas as pd

from .data_quality import (
    build_data_quality_report,
    format_quality_summary,
    save_data_quality_report,
)
from .data_loading import (
    build_dataset_inventory,
    build_metadata_frame,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from .experiment import (
    format_console_summary,
    run_model_comparison,
    save_outputs,
)
from .features import WindowConfig, build_case_feature_frame


def main() -> None:
    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    metadata = build_metadata_frame(records)
    inventory = build_dataset_inventory(records)
    quality_case_df, quality_missing_df = build_data_quality_report(records=records)
    save_data_quality_report(quality_case_df, quality_missing_df)
    inventory.to_csv("outputs/dataset_inventory.csv", index=False, encoding="utf-8-sig")

    window_config = WindowConfig()
    feature_frames = []
    for record in records:
        cleaned_signal = load_clean_signal_frame(record, common_signal_columns)
        feature_frames.append(
            build_case_feature_frame(record, cleaned_signal, window_config)
        )

    feature_df = pd.concat(feature_frames, ignore_index=True)
    results = run_model_comparison(feature_df)
    save_outputs(results)

    print("数据概览:")
    print(
        metadata[
            ["case_id", "file_name", "wind_speed", "rpm", "is_labeled"]
        ].to_string(index=False)
    )
    print()
    print("数据质量概览:")
    print(format_quality_summary(quality_case_df))
    print()
    print(format_console_summary(results))
