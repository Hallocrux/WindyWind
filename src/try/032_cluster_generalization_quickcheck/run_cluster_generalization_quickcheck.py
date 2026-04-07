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

TRY_NAME = "032_cluster_generalization_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
MECHANISM_CLUSTER_PATH = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "030_case_mechanism_clustering"
    / "case_embedding.csv"
)
TARGET_CASE_IDS = [1, 3, 6, 8, 15, 16, 17, 18]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 16
    patience: int = 3
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
    parser = argparse.ArgumentParser(description="快速验证机制簇内 / 跨簇泛化。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--matched-repeats", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mechanism_cluster_map = load_mechanism_clusters()
    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    results_df = run_generalization_check(
        dataset=dataset,
        target_case_ids=TARGET_CASE_IDS,
        mechanism_cluster_map=mechanism_cluster_map,
        train_config=train_config,
        random_seed=args.random_seed,
        matched_repeats=args.matched_repeats,
    )
    summary_df = summarize_results(results_df)

    results_df.to_csv(output_dir / "case_level_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", results_df, summary_df)

    print("机制簇内 / 跨簇泛化快速验证已完成。")
    print(f"输出目录: {output_dir}")


def load_mechanism_clusters() -> dict[int, int]:
    df = pd.read_csv(MECHANISM_CLUSTER_PATH)
    return {int(row.case_id): int(row.cluster_id) for row in df.itertuples()}


def run_generalization_check(
    dataset,
    target_case_ids: list[int],
    mechanism_cluster_map: dict[int, int],
    train_config: TrainConfig,
    random_seed: int,
    matched_repeats: int,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    windows = dataset.windows[labeled_mask]
    y_all = meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    case_values = meta["case_id"].to_numpy(dtype=int, copy=False)
    device = torch.device("cpu")

    rows: list[dict[str, object]] = []
    for case_id in target_case_ids:
        eval_cluster = mechanism_cluster_map[case_id]
        valid_idx = np.flatnonzero(case_values == case_id)
        train_idx_all = np.flatnonzero(case_values != case_id)
        candidate_train_case_ids = sorted(set(case_values[train_idx_all].tolist()))
        same_cases = [cid for cid in candidate_train_case_ids if mechanism_cluster_map[cid] == eval_cluster]
        cross_cases = [cid for cid in candidate_train_case_ids if mechanism_cluster_map[cid] != eval_cluster]
        matched_case_count = min(len(same_cases), len(cross_cases))

        train_variants = [("full", candidate_train_case_ids, 0)]
        rng = np.random.default_rng(random_seed + case_id * 100)
        for repeat in range(matched_repeats):
            same_sample = sorted(rng.choice(same_cases, size=matched_case_count, replace=False).tolist())
            cross_sample = sorted(rng.choice(cross_cases, size=matched_case_count, replace=False).tolist())
            train_variants.append(("same_cluster_matched", same_sample, repeat))
            train_variants.append(("cross_cluster_matched", cross_sample, repeat))

        for variant_name, train_case_ids, repeat_id in train_variants:
            train_mask = np.isin(case_values, np.array(train_case_ids, dtype=int))
            train_idx = np.flatnonzero(train_mask)
            pred_mean = train_and_predict_case(
                windows=windows,
                y_all=y_all,
                meta=meta,
                train_idx=train_idx,
                valid_idx=valid_idx,
                train_config=train_config,
                seed=random_seed + case_id + repeat_id,
                device=device,
            )
            true_wind_speed = float(meta.iloc[valid_idx]["wind_speed"].iloc[0])
            rows.append(
                {
                    "case_id": case_id,
                    "file_name": str(meta.iloc[valid_idx]["file_name"].iloc[0]),
                    "mechanism_cluster_id": eval_cluster,
                    "true_wind_speed": true_wind_speed,
                    "pred_mean": pred_mean,
                    "abs_error": abs(pred_mean - true_wind_speed),
                    "variant_name": variant_name,
                    "repeat_id": repeat_id,
                    "train_case_count": len(train_case_ids),
                    "train_case_ids": ",".join(str(v) for v in train_case_ids),
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "variant_name", "repeat_id"]).reset_index(drop=True)


def train_and_predict_case(
    windows: np.ndarray,
    y_all: np.ndarray,
    meta: pd.DataFrame,
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


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("variant_name", as_index=False)
        .agg(
            case_mae=("abs_error", "mean"),
            case_max_error=("abs_error", "max"),
            mean_train_case_count=("train_case_count", "mean"),
        )
        .sort_values(["case_mae", "case_max_error"])
        .reset_index(drop=True)
    )
    return summary


def write_summary_markdown(output_path: Path, results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# 机制簇内 / 跨簇泛化快速验证结论",
        "",
        f"- 目标工况：`{TARGET_CASE_IDS}`",
        "",
        "## 总览",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_max_error=`{row['case_max_error']:.4f}`, mean_train_case_count=`{row['mean_train_case_count']:.1f}`"
        )

    lines.extend(["", "## 每工况结果", ""])
    for case_id in TARGET_CASE_IDS:
        block = (
            results_df[results_df["case_id"] == case_id]
            .groupby("variant_name", as_index=False)
            .agg(abs_error_mean=("abs_error", "mean"), train_case_count=("train_case_count", "mean"))
            .sort_values("variant_name")
        )
        lines.append(f"### 工况{case_id}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: abs_error_mean=`{row['abs_error_mean']:.4f}`, train_case_count=`{row['train_case_count']:.1f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
