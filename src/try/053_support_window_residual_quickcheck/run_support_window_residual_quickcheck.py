from __future__ import annotations

import argparse
import json
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

from phase3_end_to_end_lib import build_raw_window_dataset
from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

TRY_NAME = "053_support_window_residual_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY052_CASE_PATH = REPO_ROOT / "outputs" / "try" / "052_tcn_embedding_window_signal_quickcheck" / "case_level_predictions.csv"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
HOLDOUT_CASE_IDS = [1, 3, 17, 18, 21, 22, 23, 24]
WINDOW_CONFIGS = {
    "2s": WindowConfig(sampling_rate=50.0, window_size=100, step_size=50),
    "8s": WindowConfig(sampling_rate=50.0, window_size=400, step_size=200),
}
BASELINE_VARIANTS = [
    "rpm_knn4",
    "rpm_knn4__plus__embedding_residual_knn4_concat_2s_8s_w0.5",
    "rpm_knn4__plus__embedding_residual_knn4_2s_w0.5",
    "rpm_knn4__plus__embedding_residual_knn4_8s_w0.5",
]
SUPPORT_K = 4
WINDOW_K = 4
EPS = 1e-6


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: int = 32


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
    parser = argparse.ArgumentParser(description="support-window residual quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    baseline_case_df = load_baseline_cases()

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_added_records()
    all_records = sorted([*final_records, *added_records], key=lambda record: record.case_id)
    record_by_case_id = {record.case_id: record for record in all_records}
    holdout_records = [record_by_case_id[case_id] for case_id in HOLDOUT_CASE_IDS]
    domain_by_case_id = {case_id: ("final_focus" if case_id < 21 else "added_focus") for case_id in HOLDOUT_CASE_IDS}

    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    support_case_rows: list[dict[str, object]] = []
    support_neighbor_rows: list[dict[str, object]] = []
    for holdout in holdout_records:
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        fold_case_rows, fold_neighbor_rows = run_holdout_fold(
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            train_config=train_config,
            seed=args.random_seed + holdout.case_id,
            checkpoint_dir=ckpt_dir,
            force_retrain=args.force_retrain,
            domain_name=domain_by_case_id[holdout.case_id],
        )
        support_case_rows.extend(fold_case_rows)
        support_neighbor_rows.extend(fold_neighbor_rows)

    support_case_df = pd.DataFrame(support_case_rows)
    support_neighbor_df = pd.DataFrame(support_neighbor_rows)
    merged_case_df = pd.concat([baseline_case_df, support_case_df], ignore_index=True)
    summary_df = build_summary_by_domain(merged_case_df)

    merged_case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    support_neighbor_df.to_csv(output_dir / "support_window_neighbors.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, merged_case_df)

    best_focus = summary_df.loc[summary_df["domain"] == "focus_all"].iloc[0]
    print("053 support-window residual quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best focus_all: {best_focus['variant_name']} | case_mae={best_focus['case_mae']:.4f}")


def load_baseline_cases() -> pd.DataFrame:
    case_df = pd.read_csv(TRY052_CASE_PATH, encoding="utf-8-sig")
    return case_df.loc[case_df["variant_name"].isin(BASELINE_VARIANTS)].copy().reset_index(drop=True)


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
    checkpoint_dir: Path,
    force_retrain: bool,
    domain_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_pred, reference_records = predict_rpm_knn4_with_neighbors(train_records, float(holdout.rpm))

    per_window: dict[str, dict[str, object]] = {}
    for order, (window_label, window_config) in enumerate(WINDOW_CONFIGS.items(), start=1):
        fold_seed = seed + order * 1000
        result = load_or_train_fold(
            train_records=train_records,
            holdout=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            window_label=window_label,
            train_config=train_config,
            seed=fold_seed,
            checkpoint_dir=checkpoint_dir,
            force_retrain=force_retrain,
        )
        per_window[window_label] = result

    case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for window_label in ("2s", "8s"):
        pred, rows = predict_support_window_residual(
            train_records=train_records,
            holdout=holdout,
            reference_records=reference_records,
            base_pred=base_pred,
            fold_data=per_window[window_label],
            variant_name=f"rpm_knn4__plus__support_window_residual_{window_label}_w0.5",
        )
        case_rows.append(build_case_row(domain_name, f"rpm_knn4__plus__support_window_residual_{window_label}_w0.5", holdout, pred))
        neighbor_rows.extend(rows)

    pred_avg = 0.5 * (
        extract_pred(case_rows, "rpm_knn4__plus__support_window_residual_2s_w0.5")
        + extract_pred(case_rows, "rpm_knn4__plus__support_window_residual_8s_w0.5")
    )
    case_rows.append(build_case_row(domain_name, "rpm_knn4__plus__support_window_residual_avg_2s_8s_w0.5", holdout, pred_avg))

    pred_concat, concat_rows = predict_support_window_residual_concat(
        train_records=train_records,
        holdout=holdout,
        reference_records=reference_records,
        base_pred=base_pred,
        fold_data_2s=per_window["2s"],
        fold_data_8s=per_window["8s"],
        variant_name="rpm_knn4__plus__support_window_residual_concat_2s_8s_w0.5",
    )
    case_rows.append(build_case_row(domain_name, "rpm_knn4__plus__support_window_residual_concat_2s_8s_w0.5", holdout, pred_concat))
    neighbor_rows.extend(concat_rows)
    return case_rows, neighbor_rows


def load_or_train_fold(
    *,
    train_records: list[DatasetRecord],
    holdout: DatasetRecord,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    window_label: str,
    train_config: TrainConfig,
    seed: int,
    checkpoint_dir: Path,
    force_retrain: bool,
) -> dict[str, object]:
    train_dataset = build_raw_window_dataset(
        train_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in train_records},
        window_config,
    )
    eval_dataset = build_raw_window_dataset(
        [holdout],
        {holdout.case_id: cleaned_signal_frames[holdout.case_id]},
        window_config,
    )
    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows

    ckpt_base = checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}"
    ckpt_path = ckpt_base.with_suffix(".pt")
    norm_path = checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}_norm.npz"
    meta_path = checkpoint_dir / f"fold_case_{holdout.case_id}_{window_label}.json"
    X_train_norm, X_eval_norm, mean, std = normalize_windows_by_channel(X_train, X_eval)

    if ckpt_path.exists() and norm_path.exists() and meta_path.exists() and not force_retrain:
        model = TinyTCNEncoderRegressor(in_channels=X_train.shape[1])
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        norm = np.load(norm_path)
        mean = norm["mean"]
        std = norm["std"]
        X_train_norm = ((X_train - mean) / std).astype(np.float32)
        X_eval_norm = ((X_eval - mean) / std).astype(np.float32)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu")
        model = TinyTCNEncoderRegressor(in_channels=X_train.shape[1]).to(device)
        train_model(model, X_train_norm, y_train, train_config, device)
        torch.save(model.state_dict(), ckpt_path)
        np.savez(norm_path, mean=mean, std=std)
        meta_path.write_text(
            json.dumps(
                {
                    "holdout_case_id": holdout.case_id,
                    "window_label": window_label,
                    "seed": seed,
                    "window_size": window_config.window_size,
                    "step_size": window_config.step_size,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = model.cpu()

    with torch.no_grad():
        train_tensor = torch.from_numpy(X_train_norm).float()
        eval_tensor = torch.from_numpy(X_eval_norm).float()
        train_embedding = model.encode(train_tensor).cpu().numpy()
        eval_embedding = model.encode(eval_tensor).cpu().numpy()

    train_window_df = train_dataset.meta_df.copy()
    eval_window_df = eval_dataset.meta_df.copy()
    train_case_df = build_case_table(train_records)
    train_case_df = add_rpm_oof_predictions(train_case_df)
    residual_lookup = train_case_df.set_index("case_id")["rpm_residual_oof"]
    train_window_df["embedding_index"] = np.arange(len(train_window_df))
    train_window_df["support_residual"] = train_window_df["case_id"].map(residual_lookup).astype(float)
    eval_window_df["embedding_index"] = np.arange(len(eval_window_df))
    return {
        "train_window_df": train_window_df.reset_index(drop=True),
        "eval_window_df": eval_window_df.reset_index(drop=True),
        "train_embedding": train_embedding,
        "eval_embedding": eval_embedding,
        "train_case_df": train_case_df,
    }


def predict_support_window_residual(
    *,
    train_records: list[DatasetRecord],
    holdout: DatasetRecord,
    reference_records: list[DatasetRecord],
    base_pred: float,
    fold_data: dict[str, object],
    variant_name: str,
) -> tuple[float, list[dict[str, object]]]:
    train_window_df = fold_data["train_window_df"]
    support_case_ids = {record.case_id for record in reference_records}
    support_df = train_window_df.loc[train_window_df["case_id"].isin(support_case_ids)].reset_index(drop=True)
    support_indices = support_df["embedding_index"].to_numpy(dtype=int, copy=False)
    support_embedding = fold_data["train_embedding"][support_indices]
    eval_embedding = fold_data["eval_embedding"]

    window_preds: list[float] = []
    neighbor_rows: list[dict[str, object]] = []
    for window_order, eval_vec in enumerate(eval_embedding):
        pred, rows = predict_window_residual(
            support_df=support_df,
            support_embedding=support_embedding,
            eval_vec=eval_vec,
            holdout=holdout,
            window_order=window_order,
            variant_name=variant_name,
        )
        window_preds.append(pred)
        neighbor_rows.extend(rows)

    clip_value = max(float(fold_data["train_case_df"]["rpm_residual_oof"].abs().quantile(0.95)), 0.15)
    case_residual = float(np.clip(np.mean(window_preds), -clip_value, clip_value))
    pred = float(base_pred + 0.5 * case_residual)
    return pred, neighbor_rows


def predict_support_window_residual_concat(
    *,
    train_records: list[DatasetRecord],
    holdout: DatasetRecord,
    reference_records: list[DatasetRecord],
    base_pred: float,
    fold_data_2s: dict[str, object],
    fold_data_8s: dict[str, object],
    variant_name: str,
) -> tuple[float, list[dict[str, object]]]:
    support_case_ids = {record.case_id for record in reference_records}
    support_df_2s = fold_data_2s["train_window_df"].loc[fold_data_2s["train_window_df"]["case_id"].isin(support_case_ids)].copy()
    support_df_8s = fold_data_8s["train_window_df"].loc[fold_data_8s["train_window_df"]["case_id"].isin(support_case_ids)].copy()

    support_case_table = (
        support_df_2s.groupby("case_id", as_index=False)
        .agg(residual_target=("support_residual", "first"))
        .merge(
            support_df_8s.groupby("case_id", as_index=False).agg(dummy=("support_residual", "first")),
            on="case_id",
            how="inner",
        )
    )
    support_case_ids_order = support_case_table["case_id"].to_numpy(dtype=int, copy=False)
    case_embedding_2s = build_case_embedding_from_windows(
        support_df_2s,
        fold_data_2s["train_embedding"],
        support_case_ids_order,
    )
    case_embedding_8s = build_case_embedding_from_windows(
        support_df_8s,
        fold_data_8s["train_embedding"],
        support_case_ids_order,
    )
    support_matrix = np.concatenate([case_embedding_2s, case_embedding_8s], axis=1)
    eval_vec = np.concatenate(
        [
            fold_data_2s["eval_embedding"].mean(axis=0),
            fold_data_8s["eval_embedding"].mean(axis=0),
        ]
    )
    residual_target = support_case_table["residual_target"].to_numpy(dtype=float, copy=False)
    pred_residual, rows = predict_case_residual_from_case_support(
        support_case_ids=support_case_ids_order,
        support_matrix=support_matrix,
        residual_target=residual_target,
        eval_vec=eval_vec,
        holdout=holdout,
        variant_name=variant_name,
    )
    clip_value = max(float(fold_data_2s["train_case_df"]["rpm_residual_oof"].abs().quantile(0.95)), 0.15)
    pred_residual = float(np.clip(pred_residual, -clip_value, clip_value))
    pred = float(base_pred + 0.5 * pred_residual)
    return pred, rows


def build_case_embedding_from_windows(
    support_df: pd.DataFrame,
    train_embedding: np.ndarray,
    support_case_ids_order: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for case_id in support_case_ids_order:
        block = support_df.loc[support_df["case_id"] == int(case_id)]
        indices = block["embedding_index"].to_numpy(dtype=int, copy=False)
        rows.append(train_embedding[indices].mean(axis=0))
    return np.vstack(rows)


def predict_case_residual_from_case_support(
    *,
    support_case_ids: np.ndarray,
    support_matrix: np.ndarray,
    residual_target: np.ndarray,
    eval_vec: np.ndarray,
    holdout: DatasetRecord,
    variant_name: str,
) -> tuple[float, list[dict[str, object]]]:
    mean = support_matrix.mean(axis=0, keepdims=True)
    std = support_matrix.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    support_scaled = (support_matrix - mean) / std
    eval_scaled = (np.asarray(eval_vec, dtype=float)[None, :] - mean) / std
    distances = np.sqrt(np.sum(np.square(support_scaled - eval_scaled), axis=1))
    k = min(SUPPORT_K, len(distances))
    order = np.argsort(distances)[:k]
    weights = 1.0 / np.maximum(distances[order], EPS)
    pred = float(np.average(residual_target[order], weights=weights))

    rows: list[dict[str, object]] = []
    weight_sum = weights.sum()
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "variant_name": variant_name,
                "holdout_case_id": holdout.case_id,
                "holdout_file_name": holdout.file_name,
                "window_index": -1,
                "neighbor_rank": rank,
                "neighbor_case_id": int(support_case_ids[idx]),
                "neighbor_window_index": -1,
                "distance": float(distances[idx]),
                "weight": float(weights[rank - 1] / weight_sum),
                "neighbor_residual_target": float(residual_target[idx]),
            }
        )
    return pred, rows


def predict_window_residual(
    *,
    support_df: pd.DataFrame,
    support_embedding: np.ndarray,
    eval_vec: np.ndarray,
    holdout: DatasetRecord,
    window_order: int,
    variant_name: str,
) -> tuple[float, list[dict[str, object]]]:
    mean = support_embedding.mean(axis=0, keepdims=True)
    std = support_embedding.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    support_scaled = (support_embedding - mean) / std
    eval_scaled = (np.asarray(eval_vec, dtype=float)[None, :] - mean) / std
    distances = np.sqrt(np.sum(np.square(support_scaled - eval_scaled), axis=1))
    k = min(WINDOW_K, len(distances))
    order = np.argsort(distances)[:k]
    weights = 1.0 / np.maximum(distances[order], EPS)
    target = support_df["support_residual"].to_numpy(dtype=float, copy=False)
    pred = float(np.average(target[order], weights=weights))

    rows: list[dict[str, object]] = []
    weight_sum = weights.sum()
    for rank, idx in enumerate(order, start=1):
        row = support_df.iloc[idx]
        rows.append(
            {
                "variant_name": variant_name,
                "holdout_case_id": holdout.case_id,
                "holdout_file_name": holdout.file_name,
                "window_index": window_order,
                "neighbor_rank": rank,
                "neighbor_case_id": int(row["case_id"]),
                "neighbor_window_index": int(row["window_index"]),
                "distance": float(distances[idx]),
                "weight": float(weights[rank - 1] / weight_sum),
                "neighbor_residual_target": float(row["support_residual"]),
            }
        )
    return pred, rows


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


def normalize_windows_by_channel(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    X_train_norm = ((X_train - channel_mean) / channel_std).astype(np.float32)
    X_eval_norm = ((X_eval - channel_mean) / channel_std).astype(np.float32)
    return X_train_norm, X_eval_norm, channel_mean.astype(np.float32), channel_std.astype(np.float32)


def build_case_table(records: list[DatasetRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "rpm": float(record.rpm),
            }
            for record in records
        ]
    ).sort_values("case_id").reset_index(drop=True)


def add_rpm_oof_predictions(train_case_df: pd.DataFrame) -> pd.DataFrame:
    preds: list[float] = []
    for case_id in train_case_df["case_id"]:
        inner = train_case_df.loc[train_case_df["case_id"] != case_id].copy()
        rpm_value = float(train_case_df.loc[train_case_df["case_id"] == case_id, "rpm"].iloc[0])
        preds.append(predict_rpm_knn4_from_case_df(inner, rpm_value))
    result = train_case_df.copy()
    result["rpm_pred_oof"] = np.asarray(preds, dtype=float)
    result["rpm_residual_oof"] = result["true_wind_speed"] - result["rpm_pred_oof"]
    return result


def predict_rpm_knn4_with_neighbors(records: list[DatasetRecord], rpm_value: float) -> tuple[float, list[DatasetRecord]]:
    ordered = sorted(records, key=lambda record: abs(float(record.rpm) - rpm_value))
    neighbors = ordered[: min(SUPPORT_K, len(ordered))]
    distances = np.asarray([abs(float(record.rpm) - rpm_value) for record in neighbors], dtype=float)
    weights = 1.0 / np.maximum(distances, EPS)
    pred = np.average(np.asarray([float(record.wind_speed) for record in neighbors], dtype=float), weights=weights)
    return float(pred), neighbors


def predict_rpm_knn4_from_case_df(train_df: pd.DataFrame, rpm_value: float) -> float:
    block = train_df.assign(rpm_distance=(train_df["rpm"] - rpm_value).abs()).nsmallest(min(SUPPORT_K, len(train_df)), "rpm_distance")
    distances = block["rpm_distance"].to_numpy(dtype=float, copy=False)
    weights = 1.0 / np.maximum(distances, EPS)
    target_column = "wind_speed" if "wind_speed" in block.columns else "true_wind_speed"
    pred = np.average(block[target_column].to_numpy(dtype=float, copy=False), weights=weights)
    return float(pred)


def extract_pred(case_rows: list[dict[str, object]], variant_name: str) -> float:
    for row in case_rows:
        if row["variant_name"] == variant_name:
            return float(row["pred_wind_speed"])
    raise KeyError(variant_name)


def build_case_row(domain_name: str, variant_name: str, record: DatasetRecord, pred_wind_speed: float) -> dict[str, object]:
    signed_error = float(pred_wind_speed - float(record.wind_speed))
    return {
        "domain": domain_name,
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
        ("final_focus", case_level_df.loc[case_level_df["domain"] == "final_focus"].copy()),
        ("added_focus", case_level_df.loc[case_level_df["domain"] == "added_focus"].copy()),
        ("focus_all", case_level_df.copy()),
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
        "# support-window residual quickcheck",
        "",
        f"- holdout 工况：`{', '.join(str(case_id) for case_id in HOLDOUT_CASE_IDS)}`",
        "",
        "## 三桶汇总",
        "",
    ]
    for domain_name in ("final_focus", "added_focus", "focus_all"):
        lines.append(f"### {domain_name}")
        lines.append("")
        block = summary_df.loc[summary_df["domain"] == domain_name].copy()
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    lines.extend(["## 每工况重点对照", ""])
    for case_id, block in case_level_df.groupby("case_id", sort=True):
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
