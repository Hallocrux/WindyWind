from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase1_feature_groups_lib import (
    OUTPUT_ROOT,
    Phase1RuntimeConfig,
    build_feature_frame,
    build_feature_manifest,
    create_case_delta_heatmap,
    create_case_mae_bar,
    evaluate_promotion,
    predict_unlabeled_with_group,
    run_finalist_round,
    run_screening_round,
    select_finalists,
    write_summary_markdown,
)

DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第一阶段特征组筛选实验。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/009_phase1_feature_groups。",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="full",
        help="运行模式：dev 使用固定小数据集，full 使用全部工况。",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        help="显式指定参与实验的 case_id 列表；指定后优先级高于 --mode。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="外层并行 worker 数；>1 时并行执行 group/model 任务。",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="RandomForestRegressor 的 n_jobs 设置。",
    )
    return parser.parse_args()


def select_records(args: argparse.Namespace):
    records = scan_dataset_records()
    if args.case_ids:
        selected = set(args.case_ids)
        return [record for record in records if record.case_id in selected]
    if args.mode == "dev":
        return [record for record in records if record.case_id in set(DEV_CASE_IDS)]
    return records


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = Phase1RuntimeConfig(
        max_workers=max(1, args.max_workers),
        rf_n_jobs=args.rf_n_jobs,
    )

    records = select_records(args)
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    feature_df = build_feature_frame(
        records=records,
        cleaned_signal_frames=cleaned_signal_frames,
        config=WindowConfig(),
    )

    feature_manifest = build_feature_manifest(feature_df)
    screening_summary, case_level_screening = run_screening_round(feature_df, runtime_config)
    finalists = select_finalists(screening_summary)
    finalist_model_summary, finalist_case_level = run_finalist_round(
        feature_df,
        finalists,
        runtime_config,
    )
    unlabeled_predictions = (
        predict_unlabeled_with_group(
            feature_df,
            str(finalists.iloc[0]["group_name"]),
            runtime_config,
        )
        if not finalists.empty
        else predict_unlabeled_with_group(feature_df, "G0_BASE", runtime_config)
    )
    promotion = evaluate_promotion(
        screening_summary=screening_summary,
        screening_cases=case_level_screening,
        finalists=finalists,
        finalist_summary=finalist_model_summary,
        unlabeled_prediction=(
            float(unlabeled_predictions["predicted_wind_speed"].iloc[0])
            if not unlabeled_predictions.empty
            else None
        ),
    )

    screening_summary.to_csv(
        output_dir / "feature_group_summary.csv", index=False, encoding="utf-8-sig"
    )
    case_level_screening.to_csv(
        output_dir / "case_level_screening.csv", index=False, encoding="utf-8-sig"
    )
    finalist_model_summary.to_csv(
        output_dir / "finalist_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    finalist_case_level.to_csv(
        output_dir / "finalist_case_level_predictions.csv", index=False, encoding="utf-8-sig"
    )
    feature_manifest.to_csv(
        output_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig"
    )
    create_case_mae_bar(screening_summary, output_dir / "phase1_case_mae_bar.png")
    create_case_delta_heatmap(case_level_screening, output_dir / "phase1_case_delta_heatmap.png")
    write_summary_markdown(
        output_dir=output_dir,
        screening_summary=screening_summary,
        finalists=finalists,
        finalist_summary=finalist_model_summary,
        promotion=promotion,
        unlabeled_predictions=unlabeled_predictions,
    )

    print("第一阶段探索已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"晋升结果: {'yes' if promotion.promoted else 'no'}")


if __name__ == "__main__":
    main()
