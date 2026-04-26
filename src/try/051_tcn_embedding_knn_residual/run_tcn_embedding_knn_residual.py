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
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase3_end_to_end_lib import build_raw_window_dataset
from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

TRY_NAME = "051_tcn_embedding_knn_residual"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=100, step_size=50)
K_NEIGHBORS = 4
EPS = 1e-6


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


class TinyTCNEncoderRegressor(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(in_channels, 16, dilation=1),
            TemporalBlock(16, 32, dilation=2),
            TemporalBlock(32, 32, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(x)
        pooled = self.pool(hidden)
        return pooled.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(x)
        return self.head(embedding).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyTCN embedding kNN residual quickcheck。")
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

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_added_records()
    all_records = sorted([*final_records, *added_records], key=lambda record: record.case_id)
    domain_by_case_id = {
        **{record.case_id: "final" for record in final_records},
        **{record.case_id: "added" for record in added_records},
    }

    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for holdout in all_records:
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        fold_result = run_holdout_fold(
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            train_config=train_config,
            seed=args.random_seed + holdout.case_id,
            domain_by_case_id=domain_by_case_id,
        )
        case_rows.extend(fold_result["case_rows"])
        neighbor_rows.extend(fold_result["neighbor_rows"])

    case_level_df = pd.DataFrame(case_rows)
    neighbor_df = pd.DataFrame(neighbor_rows)
    summary_df = build_summary_by_domain(case_level_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(output_dir / "nearest_neighbors.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, case_level_df)

    best_all = summary_df.loc[summary_df["domain"] == "all_labeled"].iloc[0]
    print("051 TinyTCN embedding kNN residual quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best all_labeled: {best_all['variant_name']} | case_mae={best_all['case_mae']:.4f}")


def load_added_records() -> list[DatasetRecord]:
    manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        wind_speed = float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None
        rpm = float(row["rpm"]) if not pd.isna(row["rpm"]) else None
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=ADDED_DATA_DIR / f"工况{case_id}.csv",
                wind_speed=wind_speed,
                rpm=rpm,
                is_labeled=wind_speed is not None and rpm is not None,
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return [record for record in records if record.is_labeled]


def run_holdout_fold(
    *,
    train_records: list[DatasetRecord],
    holdout: DatasetRecord,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    train_config: TrainConfig,
    seed: int,
    domain_by_case_id: dict[int, str],
) -> dict[str, list[dict[str, object]]]:
    train_dataset = build_raw_window_dataset(
        train_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in train_records},
        WINDOW_CONFIG,
    )
    eval_dataset = build_raw_window_dataset(
        [holdout],
        {holdout.case_id: cleaned_signal_frames[holdout.case_id]},
        WINDOW_CONFIG,
    )

    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows
    X_train_norm, X_eval_norm = normalize_windows_by_channel(X_train, X_eval)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = TinyTCNEncoderRegressor(in_channels=X_train.shape[1]).to(device)
    train_model(model, X_train_norm, y_train, train_config, device)

    train_tensor = torch.from_numpy(X_train_norm).float().to(device)
    eval_tensor = torch.from_numpy(X_eval_norm).float().to(device)
    with torch.no_grad():
        train_window_pred = model(train_tensor).cpu().numpy()
        eval_window_pred = model(eval_tensor).cpu().numpy()
        train_embedding = model.encode(train_tensor).cpu().numpy()
        eval_embedding = model.encode(eval_tensor).cpu().numpy()

    train_case_df = aggregate_case_level(
        meta_df=train_dataset.meta_df,
        window_predictions=train_window_pred,
        window_embeddings=train_embedding,
    )
    eval_case_df = aggregate_case_level(
        meta_df=eval_dataset.meta_df,
        window_predictions=eval_window_pred,
        window_embeddings=eval_embedding,
    )

    holdout_embedding = eval_case_df["embedding"].iloc[0]
    direct_pred = float(eval_case_df["pred_direct"].iloc[0])
    base_pred = predict_rpm_knn4(train_records, holdout.rpm)

    train_case_df = add_rpm_oof_predictions(train_case_df)
    residual_clip = max(float(train_case_df["rpm_residual_oof"].abs().quantile(0.95)), 0.15)

    embed_pred, embed_neighbors = predict_embedding_knn(
        train_case_df=train_case_df,
        eval_embedding=holdout_embedding,
        target_column="true_wind_speed",
        neighbor_tag="embedding_knn4",
        holdout_case_id=holdout.case_id,
    )
    residual_pred, residual_neighbors = predict_embedding_knn(
        train_case_df=train_case_df,
        eval_embedding=holdout_embedding,
        target_column="rpm_residual_oof",
        neighbor_tag="embedding_residual_knn4",
        holdout_case_id=holdout.case_id,
    )
    residual_pred = float(np.clip(residual_pred, -residual_clip, residual_clip))

    domain = domain_by_case_id[holdout.case_id]
    case_rows = [
        build_case_row(domain, "rpm_knn4", holdout, base_pred),
        build_case_row(domain, "tinytcn_direct_2s", holdout, direct_pred),
        build_case_row(domain, "embedding_knn4", holdout, embed_pred),
        build_case_row(domain, "rpm_knn4__plus__embedding_residual_knn4", holdout, base_pred + residual_pred),
        build_case_row(domain, "rpm_knn4__plus__embedding_residual_knn4_w0.5", holdout, base_pred + 0.5 * residual_pred),
    ]

    neighbor_rows = []
    for block in (embed_neighbors, residual_neighbors):
        for row in block:
            row["holdout_domain"] = domain
            neighbor_rows.append(row)
    return {"case_rows": case_rows, "neighbor_rows": neighbor_rows}


def aggregate_case_level(
    *,
    meta_df: pd.DataFrame,
    window_predictions: np.ndarray,
    window_embeddings: np.ndarray,
) -> pd.DataFrame:
    work_df = meta_df[["case_id", "file_name", "wind_speed", "rpm"]].copy()
    work_df["pred_direct_window"] = window_predictions.astype(float)
    grouped_rows: list[dict[str, object]] = []
    for case_id, block in work_df.groupby("case_id", sort=False):
        indices = block.index.to_numpy(dtype=int, copy=False)
        grouped_rows.append(
            {
                "case_id": int(case_id),
                "file_name": str(block["file_name"].iloc[0]),
                "true_wind_speed": float(block["wind_speed"].iloc[0]),
                "rpm": float(block["rpm"].iloc[0]),
                "pred_direct": float(block["pred_direct_window"].mean()),
                "embedding": window_embeddings[indices].mean(axis=0).astype(float),
                "window_count": int(len(block)),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values("case_id").reset_index(drop=True)


def add_rpm_oof_predictions(train_case_df: pd.DataFrame) -> pd.DataFrame:
    rpm_preds: list[float] = []
    for case_id in train_case_df["case_id"]:
        inner_train_df = train_case_df.loc[train_case_df["case_id"] != case_id].copy()
        rpm_value = float(train_case_df.loc[train_case_df["case_id"] == case_id, "rpm"].iloc[0])
        rpm_preds.append(predict_rpm_knn4_from_case_df(inner_train_df, rpm_value))
    result = train_case_df.copy()
    result["rpm_pred_oof"] = np.asarray(rpm_preds, dtype=float)
    result["rpm_residual_oof"] = result["true_wind_speed"] - result["rpm_pred_oof"]
    return result


def predict_embedding_knn(
    *,
    train_case_df: pd.DataFrame,
    eval_embedding: np.ndarray,
    target_column: str,
    neighbor_tag: str,
    holdout_case_id: int,
) -> tuple[float, list[dict[str, object]]]:
    train_matrix = np.vstack(train_case_df["embedding"].to_numpy())
    target = train_case_df[target_column].to_numpy(dtype=float, copy=False)
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    train_scaled = (train_matrix - mean) / std
    eval_scaled = (np.asarray(eval_embedding, dtype=float)[None, :] - mean) / std
    distances = np.sqrt(np.sum(np.square(train_scaled - eval_scaled), axis=1))
    k = min(K_NEIGHBORS, len(distances))
    order = np.argsort(distances)[:k]
    weights = 1.0 / np.maximum(distances[order], EPS)
    pred = float(np.average(target[order], weights=weights))

    neighbor_rows: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        neighbor_rows.append(
            {
                "variant_name": neighbor_tag,
                "holdout_case_id": holdout_case_id,
                "neighbor_rank": rank,
                "neighbor_case_id": int(train_case_df.iloc[idx]["case_id"]),
                "neighbor_file_name": str(train_case_df.iloc[idx]["file_name"]),
                "distance": float(distances[idx]),
                "weight": float(weights[rank - 1] / weights.sum()),
                "neighbor_true_wind_speed": float(train_case_df.iloc[idx]["true_wind_speed"]),
                "neighbor_rpm": float(train_case_df.iloc[idx]["rpm"]),
                "neighbor_target_value": float(target[idx]),
            }
        )
    return pred, neighbor_rows


def predict_rpm_knn4(records: list[DatasetRecord], rpm_value: float | None) -> float:
    train_df = pd.DataFrame(
        [{"rpm": float(record.rpm), "wind_speed": float(record.wind_speed)} for record in records]
    )
    return predict_rpm_knn4_from_case_df(train_df, float(rpm_value))


def predict_rpm_knn4_from_case_df(train_df: pd.DataFrame, rpm_value: float) -> float:
    block = train_df.assign(rpm_distance=(train_df["rpm"] - rpm_value).abs()).nsmallest(min(K_NEIGHBORS, len(train_df)), "rpm_distance")
    distances = block["rpm_distance"].to_numpy(dtype=float, copy=False)
    weights = 1.0 / np.maximum(distances, EPS)
    target_column = "wind_speed" if "wind_speed" in block.columns else "true_wind_speed"
    pred = np.average(block[target_column].to_numpy(dtype=float, copy=False), weights=weights)
    return float(pred)


def normalize_windows_by_channel(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    return ((X_train - channel_mean) / channel_std).astype(np.float32), ((X_eval - channel_mean) / channel_std).astype(np.float32)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
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
    X_valid_tensor = torch.from_numpy(X_train).float().to(device)
    y_valid_tensor = torch.from_numpy(y_train).float().to(device)
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def build_case_row(domain: str, variant_name: str, record: DatasetRecord, pred_wind_speed: float) -> dict[str, object]:
    signed_error = float(pred_wind_speed - float(record.wind_speed))
    return {
        "domain": domain,
        "variant_name": variant_name,
        "case_id": record.case_id,
        "file_name": record.file_name,
        "true_wind_speed": float(record.wind_speed),
        "rpm": float(record.rpm),
        "pred_wind_speed": float(pred_wind_speed),
        "signed_error": signed_error,
        "abs_error": abs(signed_error),
    }


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, subset in (
        ("final", case_level_df.loc[case_level_df["domain"] == "final"].copy()),
        ("added", case_level_df.loc[case_level_df["domain"] == "added"].copy()),
        ("all_labeled", case_level_df.copy()),
    ):
        for variant_name, block in subset.groupby("variant_name", sort=False):
            rows.append(
                {
                    "domain": domain_name,
                    "variant_name": variant_name,
                    "case_mae": float(block["abs_error"].mean()),
                    "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                    "mean_signed_error": float(block["signed_error"].mean()),
                    "case_count": int(len(block)),
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, summary_df: pd.DataFrame, case_level_df: pd.DataFrame) -> None:
    lines = [
        "# TinyTCN embedding kNN residual quickcheck",
        "",
        f"- `final` 带标签工况数：`{int((case_level_df['domain'] == 'final').sum() / case_level_df['variant_name'].nunique())}`",
        f"- `added` 带标签工况数：`{int((case_level_df['domain'] == 'added').sum() / case_level_df['variant_name'].nunique())}`",
        f"- 变体数：`{int(case_level_df['variant_name'].nunique())}`",
        "",
        "## 三桶汇总",
        "",
    ]
    for domain_name in ("final", "added", "all_labeled"):
        lines.append(f"### {domain_name}")
        lines.append("")
        block = summary_df.loc[summary_df["domain"] == domain_name].copy()
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    lines.extend(["## 关注工况", ""])
    focus_cases = case_level_df.loc[case_level_df["case_id"].isin([1, 17, 18, 22])].copy()
    for case_id, block in focus_cases.groupby("case_id", sort=True):
        lines.append(f"### 工况{int(case_id)}")
        lines.append("")
        for _, row in block.sort_values(["abs_error", "variant_name"]).iterrows():
            lines.append(
                f"- `{row['variant_name']}`: true=`{row['true_wind_speed']:.4f}`, pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
