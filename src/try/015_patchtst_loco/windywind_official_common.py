from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
for candidate in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY013_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase3_cnn_tcn_lib import (
    TorchTrainConfig,
    normalize_windows_by_channel,
    predict_torch_model_unlabeled,
    summarize_predictions,
    train_torch_model,
)
from phase3_end_to_end_lib import RawDataset, build_raw_window_dataset

DEFAULT_DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"输出目录，默认写到 {default_output_dir.as_posix()}。",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="dev",
        help="运行模式：dev 使用固定小数据集，full 使用全部工况。",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        help="显式指定参与实验的 case_id 列表；指定后优先级高于 --mode。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=42)


def select_records(args: argparse.Namespace):
    records = scan_dataset_records()
    if args.case_ids:
        selected = set(args.case_ids)
        return [record for record in records if record.case_id in selected]
    if args.mode == "dev":
        return [record for record in records if record.case_id in set(DEFAULT_DEV_CASE_IDS)]
    return records


def build_dataset_from_records(records: list) -> RawDataset:
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    return build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())


def build_train_config(args: argparse.Namespace) -> TorchTrainConfig:
    return TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def resolve_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_scalar_model_loco(
    dataset: RawDataset,
    model_builder: Callable[[int, int, torch.device], nn.Module],
    train_config: TorchTrainConfig,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = dataset.windows[labeled_mask]
    split_map = build_split_map_from_meta(labeled_meta)
    y_all = labeled_meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    predictions: list[pd.DataFrame] = []

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = resolve_device()

    for _, (train_idx, valid_idx) in split_map.items():
        X_train = labeled_windows[train_idx]
        X_valid = labeled_windows[valid_idx]
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]
        X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)

        model = model_builder(X_train.shape[1], X_train.shape[2], device).to(device)
        train_torch_model(
            model=model,
            X_train=X_train_norm,
            y_train=y_train,
            X_valid=X_valid_norm,
            y_valid=y_valid,
            config=train_config,
            device=device,
        )
        with torch.no_grad():
            pred = model(torch.from_numpy(X_valid_norm).float().to(device)).cpu().numpy()

        valid_df = labeled_meta.iloc[valid_idx][
            ["case_id", "file_name", "window_index", "start_time", "end_time", "wind_speed"]
        ].copy()
        valid_df = valid_df.rename(columns={"wind_speed": "true_wind_speed"})
        valid_df["pred_wind_speed"] = pred
        predictions.append(valid_df)

    return pd.concat(predictions, ignore_index=True)


def predict_scalar_model_unlabeled(
    dataset: RawDataset,
    model_builder: Callable[[int, int, torch.device], nn.Module],
    train_config: TorchTrainConfig,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    unlabeled_mask = dataset.meta_df["wind_speed"].isna().to_numpy()
    if not unlabeled_mask.any():
        return pd.DataFrame(columns=["case_id", "file_name", "predicted_wind_speed"])

    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    unlabeled_meta = dataset.meta_df.loc[unlabeled_mask].reset_index(drop=True)
    X_train = dataset.windows[labeled_mask]
    X_unlabeled = dataset.windows[unlabeled_mask]
    y_train = labeled_meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = resolve_device()
    X_train_norm, X_unlabeled_norm = normalize_windows_by_channel(X_train, X_unlabeled)
    model = model_builder(X_train.shape[1], X_train.shape[2], device).to(device)
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
        pred = model(torch.from_numpy(X_unlabeled_norm).float().to(device)).cpu().numpy()
    pred_df = unlabeled_meta[["case_id", "file_name"]].copy()
    pred_df["pred_wind_speed"] = pred
    return (
        pred_df.groupby(["case_id", "file_name"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "predicted_wind_speed"})
    )


def write_summary_markdown(
    output_dir: Path,
    *,
    model_name: str,
    summary_row: dict[str, object],
    records: list,
    official_repo_url: str,
    official_commit: str,
) -> None:
    lines = [
        f"# {model_name} windywind 结论",
        "",
        f"- 运行工况：`{[record.case_id for record in records]}`",
        f"- 官方仓库：`{official_repo_url}`",
        f"- 官方 commit：`{official_commit}`",
        f"- case_mae：`{float(summary_row['case_mae']):.4f}`",
        f"- case_rmse：`{float(summary_row['case_rmse']):.4f}`",
        f"- case_mape：`{float(summary_row['case_mape']):.4f}%`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_metadata(
    output_dir: Path,
    *,
    model_name: str,
    args: argparse.Namespace,
    records: list,
    official_repo_url: str,
    official_commit: str,
) -> None:
    device = resolve_device()
    payload = {
        "model_name": model_name,
        "official_repo_url": official_repo_url,
        "official_commit": official_commit,
        "mode": args.mode,
        "case_ids": [record.case_id for record in records],
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "random_seed": args.random_seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_split_map_from_meta(meta_df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    case_ids = sorted(meta_df["case_id"].unique())
    case_values = meta_df["case_id"].to_numpy(dtype=int, copy=False)
    return {
        case_id: (
            np.flatnonzero(case_values != case_id),
            np.flatnonzero(case_values == case_id),
        )
        for case_id in case_ids
    }
