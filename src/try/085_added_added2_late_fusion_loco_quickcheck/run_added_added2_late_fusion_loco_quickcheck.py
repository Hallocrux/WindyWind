from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY013_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase3_cnn_tcn_lib import TinyTCN, TorchTrainConfig, train_torch_model  # noqa: E402
from phase3_end_to_end_lib import build_raw_window_dataset  # noqa: E402
from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame  # noqa: E402
from src.current.features import WindowConfig  # noqa: E402

TRY_NAME = "085_added_added2_late_fusion_loco_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DATA_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
DEFAULT_HOLDOUT_CASE_IDS = [22, 23, 25, 29]
WINDOW_CONFIGS = {
    "2s": WindowConfig(sampling_rate=50.0, window_size=100, step_size=50),
    "8s": WindowConfig(sampling_rate=50.0, window_size=400, step_size=200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="added+added2 训练池的 2s+8s 晚融合 LOCO quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--holdout-case-ids",
        type=int,
        nargs="+",
        default=DEFAULT_HOLDOUT_CASE_IDS,
        help="只对这些工况做 LOCO holdout，训练池仍为 added+added2 的其余带标签工况。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    added_records = load_manifest_records(ADDED_MANIFEST_PATH, ADDED_DATA_DIR, "added")
    added2_records = load_manifest_records(ADDED2_MANIFEST_PATH, ADDED2_DATA_DIR, "added2")
    all_records = sorted([*added_records, *added2_records], key=lambda record: record.case_id)
    records_by_case_id = {record.case_id: record for record in all_records}
    holdout_case_ids = validate_holdout_case_ids(args.holdout_case_ids, records_by_case_id)
    holdout_records = [records_by_case_id[case_id] for case_id in holdout_case_ids]
    domain_by_case_id = {
        **{record.case_id: "added" for record in added_records},
        **{record.case_id: "added2" for record in added2_records},
    }

    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    prediction_frames: dict[str, pd.DataFrame] = {}
    for order, (window_label, window_config) in enumerate(WINDOW_CONFIGS.items(), start=1):
        prediction_frames[window_label] = run_quick_loco_for_window(
            all_records=all_records,
            holdout_records=holdout_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            train_config=train_config,
            domain_by_case_id=domain_by_case_id,
            seed=args.random_seed + order * 1000,
        )

    case_level_df = merge_predictions(prediction_frames)
    summary_df = build_summary_by_domain(case_level_df)
    write_summary_markdown(output_dir / "summary.md", case_level_df, summary_df, holdout_case_ids)
    write_run_config(output_dir / "run_config.json", args, all_records, holdout_case_ids, common_signal_columns)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")

    best_row = summary_df.iloc[0]
    print("085 added+added2 训练池的 2s+8s 晚融合 LOCO quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best: {best_row['domain']} | {best_row['model_name']} | case_mae={best_row['case_mae']:.4f}")


def load_manifest_records(manifest_path: Path, data_dir: Path, domain_name: str) -> list[DatasetRecord]:
    manifest_df = pd.read_csv(manifest_path)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        wind_speed = pd.to_numeric(pd.Series([row["wind_speed"]]), errors="coerce").iloc[0]
        rpm = pd.to_numeric(pd.Series([row["rpm"]]), errors="coerce").iloc[0]
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=data_dir / f"工况{case_id}.csv",
                wind_speed=float(wind_speed) if pd.notna(wind_speed) else None,
                rpm=float(rpm) if pd.notna(rpm) else None,
                is_labeled=pd.notna(wind_speed) and pd.notna(rpm),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=f"{domain_name}: {row['notes']}",
            )
        )
    return [record for record in records if record.is_labeled]


def validate_holdout_case_ids(case_ids: list[int], records_by_case_id: dict[int, DatasetRecord]) -> list[int]:
    unique_case_ids = []
    missing_case_ids = []
    for case_id in case_ids:
        if case_id not in records_by_case_id:
            missing_case_ids.append(case_id)
            continue
        if case_id not in unique_case_ids:
            unique_case_ids.append(case_id)
    if missing_case_ids:
        raise ValueError(f"以下 holdout case_id 不在 added+added2 带标签池中: {missing_case_ids}")
    if len(unique_case_ids) == 0:
        raise ValueError("holdout_case_ids 不能为空。")
    return unique_case_ids


def run_quick_loco_for_window(
    *,
    all_records: list[DatasetRecord],
    holdout_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    train_config: TorchTrainConfig,
    domain_by_case_id: dict[int, str],
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_index, holdout in enumerate(holdout_records):
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        pred_df = train_and_predict_holdout(
            train_records=train_records,
            eval_record=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            train_config=train_config,
            seed=seed + fold_index * 100 + holdout.case_id,
        )
        pred_df["domain"] = domain_by_case_id[holdout.case_id]
        pred_df["window_label"] = window_config.window_size
        rows.append(pred_df)
    return pd.concat(rows, ignore_index=True)


def train_and_predict_holdout(
    *,
    train_records: list[DatasetRecord],
    eval_record: DatasetRecord,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    train_config: TorchTrainConfig,
    seed: int,
) -> pd.DataFrame:
    train_dataset = build_raw_window_dataset(
        train_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in train_records},
        window_config,
    )
    eval_dataset = build_raw_window_dataset(
        [eval_record],
        {eval_record.case_id: cleaned_signal_frames[eval_record.case_id]},
        window_config,
    )

    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows
    X_train_norm, X_eval_norm, _, _ = normalize_windows_by_channel(X_train, X_eval)

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
        pred = model(torch.from_numpy(X_eval_norm).float().to(device)).cpu().numpy()

    pred_df = eval_dataset.meta_df[["case_id", "file_name", "wind_speed", "rpm"]].copy()
    pred_df["pred_wind_speed"] = pred.astype(float)
    return (
        pred_df.groupby(["case_id", "file_name", "wind_speed", "rpm"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"wind_speed": "true_wind_speed"})
    )


def normalize_windows_by_channel(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    X_train_norm = ((X_train - channel_mean) / channel_std).astype(np.float32)
    X_eval_norm = ((X_eval - channel_mean) / channel_std).astype(np.float32)
    return X_train_norm, X_eval_norm, channel_mean.astype(np.float32), channel_std.astype(np.float32)


def merge_predictions(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for window_label, frame in prediction_frames.items():
        block = frame.rename(columns={"pred_wind_speed": f"pred_{window_label}"})
        keep_columns = ["case_id", "file_name", "true_wind_speed", "rpm", "domain", f"pred_{window_label}"]
        if base is None:
            base = block[keep_columns].copy()
        else:
            base = base.merge(block[["case_id", f"pred_{window_label}"]], on="case_id", how="left", validate="one_to_one")
    assert base is not None

    base["pred_2s_8s_fusion"] = 0.5 * (base["pred_2s"] + base["pred_8s"])
    for model_name in ("2s", "8s", "2s_8s_fusion"):
        pred_col = f"pred_{model_name}"
        signed_col = f"signed_error_{model_name}"
        abs_col = f"abs_error_{model_name}"
        base[signed_col] = base[pred_col] - base["true_wind_speed"]
        base[abs_col] = base[signed_col].abs()
    return base.sort_values(["domain", "case_id"]).reset_index(drop=True)


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name in ("added", "added2", "all_holdouts"):
        subset = case_level_df if domain_name == "all_holdouts" else case_level_df.loc[case_level_df["domain"] == domain_name].copy()
        for model_name in ("2s", "8s", "2s_8s_fusion"):
            signed_error = subset[f"signed_error_{model_name}"].to_numpy(dtype=float, copy=False)
            abs_error = subset[f"abs_error_{model_name}"].to_numpy(dtype=float, copy=False)
            rows.append(
                {
                    "domain": domain_name,
                    "model_name": model_name,
                    "case_mae": float(np.mean(abs_error)),
                    "case_rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
                    "mean_signed_error": float(np.mean(signed_error)),
                    "case_count": int(len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "model_name"]).reset_index(drop=True)


def write_run_config(
    output_path: Path,
    args: argparse.Namespace,
    all_records: list[DatasetRecord],
    holdout_case_ids: list[int],
    common_signal_columns: list[str],
) -> None:
    payload = {
        "try_name": TRY_NAME,
        "date": "2026-04-09",
        "train_pool_case_ids": [record.case_id for record in all_records],
        "train_pool_case_count": len(all_records),
        "holdout_case_ids": holdout_case_ids,
        "holdout_case_count": len(holdout_case_ids),
        "window_configs": {
            label: {"window_size": cfg.window_size, "step_size": cfg.step_size}
            for label, cfg in WINDOW_CONFIGS.items()
        },
        "train_args": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "random_seed": args.random_seed,
        },
        "common_signal_column_count": len(common_signal_columns),
        "common_signal_columns": common_signal_columns,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_markdown(
    output_path: Path,
    case_level_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    holdout_case_ids: list[int],
) -> None:
    lines = [
        "# added+added2 训练池的 2s+8s 晚融合 LOCO quickcheck",
        "",
        f"- 日期：`2026-04-09`",
        f"- holdout case_ids：`{', '.join(str(case_id) for case_id in holdout_case_ids)}`",
        f"- `added` holdout 数：`{int((case_level_df['domain'] == 'added').sum())}`",
        f"- `added2` holdout 数：`{int((case_level_df['domain'] == 'added2').sum())}`",
        "",
        "## 汇总",
        "",
    ]
    for domain_name in ("added", "added2", "all_holdouts"):
        lines.append(f"### {domain_name}")
        lines.append("")
        block = summary_df.loc[summary_df["domain"] == domain_name]
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['model_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`, "
                f"case_count=`{int(row['case_count'])}`"
            )
        lines.append("")

    lines.extend(["## 每工况预测", ""])
    for _, row in case_level_df.iterrows():
        lines.append(
            f"- `{row['domain']} | 工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, "
            f"pred_2s=`{row['pred_2s']:.4f}`, pred_8s=`{row['pred_8s']:.4f}`, "
            f"pred_fusion=`{row['pred_2s_8s_fusion']:.4f}`, abs_fusion=`{row['abs_error_2s_8s_fusion']:.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
