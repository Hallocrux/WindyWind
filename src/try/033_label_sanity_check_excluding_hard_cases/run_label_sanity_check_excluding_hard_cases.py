from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "033_label_sanity_check_excluding_hard_cases"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
HARD_CASE_IDS = [1, 3, 17, 18]
VALIDATION_CASE_IDS = [15, 16, 19, 20]
SUSPICIOUS_CASE_IDS = [1, 18]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[..., : x.shape[-1]]
        out = self.relu1(out)
        out = self.conv2(out)
        out = out[..., : x.shape[-1]]
        out = self.relu2(out)
        return out + self.downsample(x)


class TinyTCN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(in_channels, 16, dilation=1),
            TemporalBlock(16, 32, dilation=2),
            TemporalBlock(32, 32, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="去掉难工况后检查可疑标签。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())

    validation_df = evaluate_case_list(dataset, VALIDATION_CASE_IDS, train_config, args.random_seed)
    suspicious_df = evaluate_case_list(dataset, SUSPICIOUS_CASE_IDS, train_config, args.random_seed)
    summary_df = build_summary(validation_df, suspicious_df)

    validation_df.to_csv(output_dir / "validation_case_predictions.csv", index=False, encoding="utf-8-sig")
    suspicious_df.to_csv(output_dir / "suspicious_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", validation_df, suspicious_df, summary_df)

    print("去掉难工况后的标签一致性快速检查已完成。")
    print(f"输出目录: {output_dir}")


def evaluate_case_list(dataset, case_ids: list[int], train_config: TrainConfig, random_seed: int) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    windows = dataset.windows[labeled_mask]
    y_all = meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    case_values = meta["case_id"].to_numpy(dtype=int, copy=False)
    device = torch.device("cpu")

    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        valid_idx = np.flatnonzero(case_values == case_id)
        train_idx = np.flatnonzero(
            (case_values != case_id) & (~np.isin(case_values, np.array(HARD_CASE_IDS, dtype=int)))
        )
        pred_mean = train_and_predict_case(
            windows=windows,
            y_all=y_all,
            train_idx=train_idx,
            valid_idx=valid_idx,
            train_config=train_config,
            seed=random_seed + case_id,
            device=device,
        )
        true_wind_speed = float(meta.iloc[valid_idx]["wind_speed"].iloc[0])
        rows.append(
            {
                "case_id": case_id,
                "file_name": str(meta.iloc[valid_idx]["file_name"].iloc[0]),
                "true_wind_speed": true_wind_speed,
                "pred_mean": pred_mean,
                "abs_error": abs(pred_mean - true_wind_speed),
                "signed_error": pred_mean - true_wind_speed,
                "train_case_count": int(len(np.unique(case_values[train_idx]))),
                "train_case_ids": ",".join(str(v) for v in sorted(np.unique(case_values[train_idx]).tolist())),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def train_and_predict_case(
    windows: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    device: torch.device,
) -> float:
    X_train = windows[train_idx]
    X_valid = windows[valid_idx]
    y_train = y_all[train_idx]
    y_valid = y_all[valid_idx]
    X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyTCN(in_channels=X_train.shape[1]).to(device)
    train_model(model, X_train_norm, y_train, X_valid_norm, y_valid, train_config, device)
    with torch.no_grad():
        pred = model(torch.from_numpy(X_valid_norm).to(device)).cpu().numpy()
    return float(np.mean(pred))


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> None:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=config.batch_size,
        shuffle=True,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    X_valid_tensor = torch.from_numpy(X_valid).float().to(device)
    y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = config.patience
    for _ in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid_tensor)
            valid_loss = float(criterion(valid_pred, y_valid_tensor).item())
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def normalize_windows_by_channel(X_train: np.ndarray, X_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    return ((X_train - channel_mean) / channel_std).astype(np.float32), ((X_valid - channel_mean) / channel_std).astype(np.float32)


def build_summary(validation_df: pd.DataFrame, suspicious_df: pd.DataFrame) -> pd.DataFrame:
    validation_mae = float(validation_df["abs_error"].mean())
    validation_rmse = float(np.sqrt(np.mean(np.square(validation_df["signed_error"]))))
    rows = [
        {
            "group_name": "validation_added_cases",
            "case_mae": validation_mae,
            "case_rmse": validation_rmse,
            "case_count": int(len(validation_df)),
        }
    ]
    for _, row in suspicious_df.iterrows():
        rows.append(
            {
                "group_name": f"suspicious_case_{int(row['case_id'])}",
                "case_mae": float(row["abs_error"]),
                "case_rmse": float(abs(row["signed_error"])),
                "case_count": 1,
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(output_path: Path, validation_df: pd.DataFrame, suspicious_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# 去掉难工况后的标签一致性快速检查",
        "",
        f"- 去掉的难工况：`{HARD_CASE_IDS}`",
        f"- 验证新增工况：`{VALIDATION_CASE_IDS}`",
        f"- 可疑工况：`{SUSPICIOUS_CASE_IDS}`",
        "",
        "## 新增验证工况",
        "",
    ]
    for _, row in validation_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, pred=`{row['pred_mean']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
        )
    lines.extend(
        [
            "",
            f"- 新增验证 case_mae=`{float(validation_df['abs_error'].mean()):.4f}`",
            "",
            "## 可疑工况",
            "",
        ]
    )
    for _, row in suspicious_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, pred=`{row['pred_mean']:.4f}`, signed_error=`{row['signed_error']:+.4f}`, abs_error=`{row['abs_error']:.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
