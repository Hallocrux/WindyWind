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

TRY_NAME = "028_input_channel_attention_tinytcn_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TARGET_CASE_IDS = [1, 3, 17, 18]
CONTROL_CASE_IDS = [6, 8, 10, 13]


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
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[..., : x.shape[-1]]
        out = self.relu1(out)
        out = self.conv2(out)
        out = out[..., : x.shape[-1]]
        out = self.relu2(out)
        return out + self.downsample(x)


class TinyTCNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 32) -> None:
        super().__init__()
        hidden1 = min(16, out_channels)
        hidden2 = out_channels
        self.blocks = nn.Sequential(
            TemporalBlock(in_channels, hidden1, dilation=1),
            TemporalBlock(hidden1, hidden2, dilation=2),
            TemporalBlock(hidden2, hidden2, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class InputChannelSE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class SingleStreamTinyTCN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.encoder = TinyTCNEncoder(in_channels=in_channels, out_channels=32)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat).squeeze(-1)


class InputChannelAttentionTinyTCN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.attn = InputChannelSE(in_channels=in_channels, reduction=4)
        self.encoder = TinyTCNEncoder(in_channels=in_channels, out_channels=32)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        feat = self.encoder(x)
        return self.head(feat).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速验证输入通道注意力 TinyTCN。")
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

    target_case_df = evaluate_case_group(
        dataset=dataset,
        case_ids=TARGET_CASE_IDS,
        train_config=train_config,
        random_seed=args.random_seed,
    )
    target_summary_df = summarize_case_df(target_case_df)
    target_case_df.to_csv(output_dir / "target_case_variant_comparison.csv", index=False, encoding="utf-8-sig")
    target_summary_df.to_csv(output_dir / "target_case_variant_summary.csv", index=False, encoding="utf-8-sig")

    attn_better = is_attention_positive(target_summary_df, target_case_df)
    if attn_better:
        control_case_df = evaluate_case_group(
            dataset=dataset,
            case_ids=CONTROL_CASE_IDS,
            train_config=train_config,
            random_seed=args.random_seed,
        )
        control_summary_df = summarize_case_df(control_case_df)
    else:
        control_case_df = pd.DataFrame(
            columns=["case_id", "file_name", "true_wind_speed", "pred_mean", "abs_error", "variant_name"]
        )
        control_summary_df = pd.DataFrame(
            columns=["variant_name", "case_mae", "case_max_error", "improved_case_count"]
        )

    control_case_df.to_csv(output_dir / "control_case_variant_comparison.csv", index=False, encoding="utf-8-sig")
    control_summary_df.to_csv(output_dir / "control_case_variant_summary.csv", index=False, encoding="utf-8-sig")

    write_summary_markdown(
        output_path=output_dir / "summary.md",
        target_case_df=target_case_df,
        target_summary_df=target_summary_df,
        control_case_df=control_case_df,
        control_summary_df=control_summary_df,
        attn_better=attn_better,
    )

    print("输入通道注意力 TinyTCN 快速验证已完成。")
    print(f"输出目录: {output_dir}")
    print(f"目标工况: {TARGET_CASE_IDS}")
    print(f"是否继续了对照工况: {attn_better}")


def evaluate_case_group(
    dataset,
    case_ids: list[int],
    train_config: TrainConfig,
    random_seed: int,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = dataset.windows[labeled_mask]
    y_all = labeled_meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    case_values = labeled_meta["case_id"].to_numpy(dtype=int, copy=False)
    device = torch.device("cpu")

    variant_rows: list[dict[str, object]] = []
    for case_id in case_ids:
        valid_idx = np.flatnonzero(case_values == case_id)
        train_idx = np.flatnonzero(case_values != case_id)

        X_train = labeled_windows[train_idx]
        X_valid = labeled_windows[valid_idx]
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]
        X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)

        for variant_name in ("SingleStreamTinyTCN", "InputChannelAttentionTinyTCN"):
            torch.manual_seed(random_seed + case_id)
            np.random.seed(random_seed + case_id)
            if variant_name == "SingleStreamTinyTCN":
                model = SingleStreamTinyTCN(in_channels=X_train.shape[1]).to(device)
            else:
                model = InputChannelAttentionTinyTCN(in_channels=X_train.shape[1]).to(device)

            train_model(
                model=model,
                X_train=X_train_norm,
                y_train=y_train,
                X_valid=X_valid_norm,
                y_valid=y_valid,
                config=train_config,
                device=device,
            )
            with torch.no_grad():
                pred = model(torch.from_numpy(X_valid_norm).to(device)).cpu().numpy()
            pred_mean = float(np.mean(pred))
            true_wind_speed = float(labeled_meta.iloc[valid_idx]["wind_speed"].iloc[0])
            variant_rows.append(
                {
                    "case_id": case_id,
                    "file_name": str(labeled_meta.iloc[valid_idx]["file_name"].iloc[0]),
                    "true_wind_speed": true_wind_speed,
                    "pred_mean": pred_mean,
                    "abs_error": abs(pred_mean - true_wind_speed),
                    "variant_name": variant_name,
                }
            )

    return pd.DataFrame(variant_rows).sort_values(["variant_name", "case_id"]).reset_index(drop=True)


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
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    criterion = nn.MSELoss()
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
    return (
        ((X_train - channel_mean) / channel_std).astype(np.float32),
        ((X_valid - channel_mean) / channel_std).astype(np.float32),
    )


def summarize_case_df(case_df: pd.DataFrame) -> pd.DataFrame:
    baseline = case_df[case_df["variant_name"] == "SingleStreamTinyTCN"][["case_id", "abs_error"]].rename(
        columns={"abs_error": "baseline_abs_error"}
    )
    merged = case_df.merge(baseline, on="case_id", how="left")
    merged["improved_vs_baseline"] = merged["abs_error"] < merged["baseline_abs_error"] - 1e-12
    summary = (
        merged.groupby("variant_name", as_index=False)
        .agg(
            case_mae=("abs_error", "mean"),
            case_max_error=("abs_error", "max"),
            improved_case_count=("improved_vs_baseline", "sum"),
        )
        .sort_values(["case_mae", "case_max_error", "variant_name"])
        .reset_index(drop=True)
    )
    return summary


def is_attention_positive(summary_df: pd.DataFrame, case_df: pd.DataFrame) -> bool:
    attn_mae = float(
        summary_df.loc[summary_df["variant_name"] == "InputChannelAttentionTinyTCN", "case_mae"].iloc[0]
    )
    single_mae = float(
        summary_df.loc[summary_df["variant_name"] == "SingleStreamTinyTCN", "case_mae"].iloc[0]
    )
    base = case_df[case_df["variant_name"] == "SingleStreamTinyTCN"][["case_id", "abs_error"]].rename(columns={"abs_error": "single_abs_error"})
    attn = case_df[case_df["variant_name"] == "InputChannelAttentionTinyTCN"][["case_id", "abs_error"]].rename(columns={"abs_error": "attn_abs_error"})
    merged = attn.merge(base, on="case_id", how="left")
    improved_count = int((merged["attn_abs_error"] < merged["single_abs_error"]).sum())
    return attn_mae < single_mae and improved_count >= 3


def write_summary_markdown(
    output_path: Path,
    target_case_df: pd.DataFrame,
    target_summary_df: pd.DataFrame,
    control_case_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    attn_better: bool,
) -> None:
    lines = [
        "# 输入通道注意力 TinyTCN 快速验证结论",
        "",
        f"- 目标难工况：`{TARGET_CASE_IDS}`",
        f"- 对照工况：`{CONTROL_CASE_IDS}`",
        "- 窗口口径：`50Hz / 5s / 2.5s`",
        "- 注意力位置：`raw input channel gate`",
        "",
        "## 目标难工况汇总",
        "",
    ]
    for _, row in target_summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_max_error=`{row['case_max_error']:.4f}`, improved_case_count=`{int(row['improved_case_count'])}`"
        )

    lines.extend(["", "## 目标难工况逐工况对比", ""])
    single_base = target_case_df[target_case_df["variant_name"] == "SingleStreamTinyTCN"][["case_id", "abs_error"]].rename(columns={"abs_error": "single_abs_error"})
    merged_target = target_case_df.merge(single_base, on="case_id", how="left")
    for case_id in TARGET_CASE_IDS:
        lines.append(f"### 工况{case_id}")
        lines.append("")
        block = merged_target[merged_target["case_id"] == case_id].sort_values("variant_name")
        for _, row in block.iterrows():
            delta = float(row["abs_error"] - row["single_abs_error"])
            lines.append(
                f"- `{row['variant_name']}`: pred=`{row['pred_mean']:.4f}`, abs_error=`{row['abs_error']:.4f}`, vs_single=`{delta:+.4f}`"
            )
        lines.append("")

    lines.extend(["## 对照工况汇总", ""])
    if attn_better and not control_summary_df.empty:
        for _, row in control_summary_df.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_max_error=`{row['case_max_error']:.4f}`, improved_case_count=`{int(row['improved_case_count'])}`"
            )
    else:
        lines.append("- 目标难工况上未达到继续条件，因此未扩到对照工况。")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 这是一轮快速可行性验证，不是全量 19 工况的定版结论。",
            "- 这里的 attention 只加在输入通道上，不与多尺度、双流或额外加权同时混用。",
            "- 如果输入通道注意力在目标难工况和对照工况上都给出稳定正信号，才值得继续进入更大范围复核。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
