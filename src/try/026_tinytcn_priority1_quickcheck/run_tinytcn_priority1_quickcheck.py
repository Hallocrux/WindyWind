from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY013_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase3_cnn_tcn_lib import (
    TorchTrainConfig,
    build_torch_model,
    normalize_windows_by_channel,
)
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "026_tinytcn_priority1_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TARGET_CASE_IDS = [1, 3, 17, 18]
WINDOW_SCAN_CASE_PATH = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "014_phase3_tcn_window_length_scan"
    / "tcn_window_scan_case_level_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速验证 TinyTCN 第一优先级方向。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录。",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=TARGET_CASE_IDS,
        help="目标工况列表。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_case_ids = sorted(set(args.case_ids))

    multiscale_case_df = build_multiscale_case_comparison(target_case_ids)
    balanced_case_df, count_df = run_case_balanced_target_folds(
        target_case_ids=target_case_ids,
        train_config=TorchTrainConfig(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
        ),
        random_seed=args.random_seed,
    )

    case_df = pd.concat([multiscale_case_df, balanced_case_df], ignore_index=True)
    case_df = case_df.sort_values(["variant_name", "case_id"]).reset_index(drop=True)
    summary_df = summarize_case_comparison(case_df)

    case_df.to_csv(
        output_dir / "variant_case_level_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        output_dir / "variant_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    count_df.to_csv(
        output_dir / "balanced_training_case_window_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        case_df=case_df,
        summary_df=summary_df,
    )

    print("TinyTCN 第一优先级快速验证已完成。")
    print(f"输出目录: {output_dir}")
    print(f"目标工况: {target_case_ids}")


def build_multiscale_case_comparison(target_case_ids: list[int]) -> pd.DataFrame:
    case_df = pd.read_csv(WINDOW_SCAN_CASE_PATH)
    target_df = case_df[
        case_df["case_id"].isin(target_case_ids)
        & case_df["window_label"].isin(["2s", "5s", "8s"])
    ].copy()

    variant_rows: list[pd.DataFrame] = []
    for window_label, variant_name in (
        ("2s", "TinyTCN_2s"),
        ("5s", "TinyTCN_5s_baseline"),
        ("8s", "TinyTCN_8s"),
    ):
        block = target_df[target_df["window_label"] == window_label].copy()
        block = block[["case_id", "file_name", "true_wind_speed", "pred_mean", "abs_error"]]
        block["variant_name"] = variant_name
        variant_rows.append(block)

    pivot = target_df.pivot_table(
        index=["case_id", "file_name", "true_wind_speed"],
        columns="window_label",
        values="pred_mean",
    ).reset_index()
    fusion_df = pivot.copy()
    fusion_df["pred_mean"] = (fusion_df["2s"] + fusion_df["8s"]) / 2.0
    fusion_df["abs_error"] = (fusion_df["pred_mean"] - fusion_df["true_wind_speed"]).abs()
    fusion_df["variant_name"] = "TinyTCN_multiscale_late_fusion_2s_8s"
    variant_rows.append(
        fusion_df[["case_id", "file_name", "true_wind_speed", "pred_mean", "abs_error", "variant_name"]]
    )

    return pd.concat(variant_rows, ignore_index=True)


def run_case_balanced_target_folds(
    target_case_ids: list[int],
    train_config: TorchTrainConfig,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())

    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = dataset.windows[labeled_mask]
    y_all = labeled_meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    case_values = labeled_meta["case_id"].to_numpy(dtype=int, copy=False)
    device = torch.device("cpu")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    case_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for case_id in target_case_ids:
        valid_idx = np.flatnonzero(case_values == case_id)
        train_idx = np.flatnonzero(case_values != case_id)

        X_train = labeled_windows[train_idx]
        X_valid = labeled_windows[valid_idx]
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]
        train_case_ids = case_values[train_idx]

        train_counts = (
            pd.Series(train_case_ids)
            .value_counts()
            .sort_index()
            .rename_axis("train_case_id")
            .reset_index(name="window_count")
        )
        train_counts["target_eval_case_id"] = case_id
        count_rows.append(train_counts)

        sample_weights = build_case_balanced_sample_weights(train_case_ids)
        X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)
        model = build_torch_model("TinyTCN", in_channels=X_train.shape[1]).to(device)
        train_torch_model_case_balanced(
            model=model,
            X_train=X_train_norm,
            y_train=y_train,
            sample_weights=sample_weights,
            X_valid=X_valid_norm,
            y_valid=y_valid,
            config=train_config,
            device=device,
        )

        with torch.no_grad():
            pred = model(torch.from_numpy(X_valid_norm).to(device)).cpu().numpy()

        pred_mean = float(np.mean(pred))
        true_wind_speed = float(labeled_meta.iloc[valid_idx]["wind_speed"].iloc[0])
        case_rows.append(
            {
                "case_id": case_id,
                "file_name": str(labeled_meta.iloc[valid_idx]["file_name"].iloc[0]),
                "true_wind_speed": true_wind_speed,
                "pred_mean": pred_mean,
                "abs_error": abs(pred_mean - true_wind_speed),
                "variant_name": "TinyTCN_5s_case_balanced",
            }
        )

    count_df = pd.concat(count_rows, ignore_index=True)
    return pd.DataFrame(case_rows), count_df


def build_case_balanced_sample_weights(train_case_ids: np.ndarray) -> np.ndarray:
    counts = pd.Series(train_case_ids).value_counts().to_dict()
    raw_weights = np.array([1.0 / counts[int(case_id)] for case_id in train_case_ids], dtype=np.float32)
    scale = float(len(raw_weights) / raw_weights.sum())
    return raw_weights * scale


def train_torch_model_case_balanced(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: TorchTrainConfig,
    device: torch.device,
) -> None:
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
            torch.from_numpy(sample_weights).float(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    X_valid_tensor = torch.from_numpy(X_valid).float().to(device)
    y_valid_tensor = torch.from_numpy(y_valid).float().to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = config.patience

    for _ in range(config.max_epochs):
        model.train()
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = torch.mean(batch_w * torch.square(pred - batch_y))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid_tensor)
            valid_loss = float(torch.mean(torch.square(valid_pred - y_valid_tensor)).item())
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def summarize_case_comparison(case_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        case_df.groupby("variant_name", as_index=False)["abs_error"]
        .agg(case_mae="mean", case_max_error="max")
        .sort_values(["case_mae", "case_max_error", "variant_name"])
        .reset_index(drop=True)
    )
    return summary


def write_summary_markdown(
    output_path: Path,
    case_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    baseline_df = case_df[case_df["variant_name"] == "TinyTCN_5s_baseline"].copy()
    lines = [
        "# TinyTCN 第一优先级快速验证结论",
        "",
        f"- 目标工况：`{sorted(case_df['case_id'].unique().tolist())}`",
        "- 基线：`TinyTCN@5s`",
        "- 多尺度代理：`2s + 8s` 工况级晚融合",
        "- 工况均衡加权：`5s TinyTCN + inverse case-window-count weighting`",
        "",
        "## 变体总览",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_max_error=`{row['case_max_error']:.4f}`"
        )

    lines.extend(["", "## 每工况对比", ""])
    for case_id in sorted(case_df["case_id"].unique()):
        lines.append(f"### 工况{case_id}")
        lines.append("")
        block = case_df[case_df["case_id"] == case_id].sort_values("variant_name")
        baseline_error = float(
            baseline_df[baseline_df["case_id"] == case_id]["abs_error"].iloc[0]
        )
        for _, row in block.iterrows():
            delta = float(row["abs_error"] - baseline_error)
            lines.append(
                f"- `{row['variant_name']}`: pred=`{row['pred_mean']:.4f}`, abs_error=`{row['abs_error']:.4f}`, vs_baseline=`{delta:+.4f}`"
            )
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "- 这是一轮针对目标难工况的快速可行性验证，不是全量 19 工况定版结论。",
            "- 多尺度方向这里只验证“晚融合代理是否有信号”；若信号为正，再考虑实现真正的多分支网络。",
            "- 工况均衡加权方向这里只改变训练 loss，不改变模型结构、窗长或评估口径。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
