from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_ROOT = REPO_ROOT / "src" / "try" / "066_reuse_embedding_domain_split"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from reuse_embedding_domain_common import (  # noqa: E402
    build_cleaned_signal_frames,
    build_record_table,
    load_source_catalog,
    load_try053_module,
)

TRY_NAME = "086_contrastive_condition_embedding"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
WINDOW_LABELS = ("2s", "8s")
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
K_NEIGHBORS = 4
EPS = 1e-6


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    max_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    embedding_dim: int = 64
    jitter_std: float = 0.02
    channel_drop_prob: float = 0.05


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


class TemporalPyramidPooling(nn.Module):
    def __init__(self, levels: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.levels = levels

    @property
    def multiplier(self) -> int:
        return int(sum(self.levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [nn.functional.adaptive_avg_pool1d(x, level).flatten(start_dim=1) for level in self.levels]
        return torch.cat(pooled, dim=1)


class ContrastiveTinyTCNEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(in_channels, 16, dilation=1),
            TemporalBlock(16, 32, dilation=2),
            TemporalBlock(32, 32, dilation=4),
        )
        self.pool = TemporalPyramidPooling((1, 2, 4))
        self.projector = nn.Sequential(
            nn.Linear(32 * self.pool.multiplier, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(x)
        pooled = self.pool(hidden)
        embedding = self.projector(pooled)
        return nn.functional.normalize(embedding, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive condition embedding quickcheck.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=30)
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

    config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )
    set_seed(args.random_seed)
    try053 = load_try053_module()
    catalog = load_source_catalog()

    final_labeled = [record for record in catalog.final_records if record.is_labeled]
    train_records = sorted([*final_labeled, *catalog.added_records], key=lambda record: record.case_id)
    export_records = sorted([*catalog.added_records, *catalog.added2_records], key=lambda record: record.case_id)
    record_df = build_record_table(catalog)
    _, cleaned_signal_frames = build_cleaned_signal_frames(sorted([*train_records, *catalog.added2_records], key=lambda record: record.case_id))

    per_window: dict[str, dict[str, object]] = {}
    for order, window_label in enumerate(WINDOW_LABELS, start=1):
        per_window[window_label] = load_or_train_window_embeddings(
            try053=try053,
            window_label=window_label,
            train_records=train_records,
            export_records=export_records,
            cleaned_signal_frames=cleaned_signal_frames,
            config=config,
            seed=args.random_seed + order * 1000,
            ckpt_dir=ckpt_dir,
            force_retrain=args.force_retrain,
        )

    embedding_case_df = build_embedding_case_table(record_df, per_window)
    embedding_columns = [column for column in embedding_case_df.columns if column.startswith("embedding_")]
    pred_df = run_added_to_added2(embedding_case_df, embedding_columns)
    summary_df = build_summary(pred_df)
    knn_df = build_knn_neighbors(embedding_case_df, embedding_columns)
    pca_df = build_pca_table(embedding_case_df, embedding_columns)

    embedding_case_df.to_csv(output_dir / "embedding_case_table.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    knn_df.to_csv(output_dir / "knn_neighbors.csv", index=False, encoding="utf-8-sig")
    pca_df.to_csv(output_dir / "embedding_pca_coords.csv", index=False, encoding="utf-8-sig")
    write_summary(output_dir / "summary.md", summary_df, pred_df, knn_df)

    best = summary_df.iloc[0]
    print("086 contrastive condition embedding quickcheck finished.")
    print(f"output_dir={output_dir}")
    print(f"best={best['variant_name']} | case_mae={best['case_mae']:.4f}")


def load_or_train_window_embeddings(
    *,
    try053,
    window_label: str,
    train_records,
    export_records,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    config: TrainConfig,
    seed: int,
    ckpt_dir: Path,
    force_retrain: bool,
) -> dict[str, object]:
    window_config = try053.WINDOW_CONFIGS[window_label]
    train_dataset = try053.build_raw_window_dataset(
        train_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in train_records},
        window_config,
    )
    export_dataset = try053.build_raw_window_dataset(
        export_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in export_records},
        window_config,
    )
    X_train = train_dataset.windows.astype(np.float32)
    X_export = export_dataset.windows.astype(np.float32)
    train_case_ids = train_dataset.meta_df["case_id"].to_numpy(dtype=np.int64, copy=False)
    X_train_norm, X_export_norm, mean, std = normalize_windows_by_channel(X_train, X_export)

    model = ContrastiveTinyTCNEncoder(in_channels=X_train.shape[1], embedding_dim=config.embedding_dim)
    ckpt_path = ckpt_dir / f"contrastive_{window_label}.pt"
    norm_path = ckpt_dir / f"contrastive_{window_label}_norm.npz"
    meta_path = ckpt_dir / f"contrastive_{window_label}.json"
    if ckpt_path.exists() and norm_path.exists() and meta_path.exists() and not force_retrain:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        norm = np.load(norm_path)
        mean = norm["mean"]
        std = norm["std"]
        X_export_norm = ((X_export - mean) / std).astype(np.float32)
    else:
        set_seed(seed)
        train_contrastive_model(model, X_train_norm, train_case_ids, config)
        torch.save(model.state_dict(), ckpt_path)
        np.savez(norm_path, mean=mean, std=std)
        meta_path.write_text(
            json.dumps(
                {
                    "window_label": window_label,
                    "seed": seed,
                    "train_case_ids": [int(record.case_id) for record in train_records],
                    "export_case_ids": [int(record.case_id) for record in export_records],
                    "objective": "supervised_contrastive_by_case_id_without_wind_speed",
                    "embedding_dim": config.embedding_dim,
                    "pooling": "temporal_pyramid_avg_levels_1_2_4",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    model.eval()
    with torch.no_grad():
        export_embedding = batched_encode(model, X_export_norm)
    return {
        "export_meta_df": export_dataset.meta_df.copy().reset_index(drop=True),
        "export_embedding": export_embedding,
    }


def train_contrastive_model(model: nn.Module, X_train: np.ndarray, case_ids: np.ndarray, config: TrainConfig) -> None:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.copy()).float(), torch.from_numpy(case_ids.copy()).long()),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    for _ in range(config.max_epochs):
        for batch_x, batch_case_ids in loader:
            view_a = augment_windows(batch_x, config)
            view_b = augment_windows(batch_x, config)
            both_x = torch.cat([view_a, view_b], dim=0)
            both_case_ids = torch.cat([batch_case_ids, batch_case_ids], dim=0)
            embedding = model(both_x)
            loss = supervised_contrastive_loss(embedding, both_case_ids, config.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def augment_windows(x: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    result = x + torch.randn_like(x) * config.jitter_std
    if config.channel_drop_prob > 0:
        keep_mask = (torch.rand(result.shape[0], result.shape[1], 1, device=result.device) > config.channel_drop_prob).float()
        result = result * keep_mask
    return result


def supervised_contrastive_loss(embedding: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = embedding @ embedding.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(EPS))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return torch.zeros((), dtype=embedding.dtype, device=embedding.device, requires_grad=True)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)[valid] / positive_count[valid].float()
    return -mean_log_prob_pos.mean()


def batched_encode(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float()
        rows.append(model(batch).cpu().numpy())
    return np.vstack(rows).astype(np.float32)


def normalize_windows_by_channel(X_train: np.ndarray, X_export: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return ((X_train - mean) / std).astype(np.float32), ((X_export - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_embedding_case_table(record_df: pd.DataFrame, per_window: dict[str, dict[str, object]]) -> pd.DataFrame:
    result = record_df.loc[record_df["raw_source_domain"].isin(["added", "added2"])].copy().sort_values("case_id").reset_index(drop=True)
    for window_label in WINDOW_LABELS:
        embedding_df = aggregate_case_embeddings(
            per_window[window_label]["export_meta_df"],
            per_window[window_label]["export_embedding"],
            f"embedding_{window_label}",
        )
        result = result.merge(embedding_df, on="case_id", how="left")
    result["embedding_concat"] = result.apply(
        lambda row: np.concatenate(
            [
                np.asarray(row["embedding_2s"], dtype=float),
                np.asarray(row["embedding_8s"], dtype=float),
            ]
        ).astype(float),
        axis=1,
    )
    matrix = np.vstack(result["embedding_concat"].to_numpy())
    embedding_df = pd.DataFrame(
        matrix,
        columns=[f"embedding_{index + 1}" for index in range(matrix.shape[1])],
        index=result.index,
    )
    return pd.concat(
        [result.drop(columns=["embedding_2s", "embedding_8s", "embedding_concat"]), embedding_df],
        axis=1,
    ).reset_index(drop=True)


def aggregate_case_embeddings(meta_df: pd.DataFrame, window_embeddings: np.ndarray, column_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, block in meta_df.groupby("case_id", sort=False):
        indices = block.index.to_numpy(dtype=int, copy=False)
        rows.append({"case_id": int(case_id), column_name: window_embeddings[indices].mean(axis=0).astype(float)})
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def run_added_to_added2(embedding_case_df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    train_df = embedding_case_df.loc[embedding_case_df["raw_source_domain"] == "added"].copy().reset_index(drop=True)
    test_df = embedding_case_df.loc[embedding_case_df["raw_source_domain"] == "added2"].copy().reset_index(drop=True)
    residual_targets = compute_internal_oof_residual_targets(train_df)
    base_pred = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))
    pred_map = {
        "rpm_knn4": base_pred,
        "contrastive_embedding_ridge": fit_predict_embedding_ridge(train_df, test_df, embedding_columns, "wind_speed"),
        "rpm_knn4_plus_contrastive_embedding_residual_ridge": base_pred
        + fit_predict_embedding_ridge(train_df, test_df, embedding_columns, residual_targets),
    }
    rows: list[dict[str, object]] = []
    for variant_name, preds in pred_map.items():
        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            pred = float(np.asarray(preds, dtype=float)[row_idx])
            signed_error = pred - float(row["wind_speed"])
            rows.append(
                {
                    "protocol": "added_to_added2",
                    "variant_name": variant_name,
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["raw_source_domain"]),
                    "true_wind_speed": float(row["wind_speed"]),
                    "rpm": float(row["rpm"]),
                    "pred_wind_speed": pred,
                    "signed_error": signed_error,
                    "abs_error": abs(signed_error),
                }
            )
    return pd.DataFrame(rows)


def fit_predict_embedding_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame, embedding_columns: list[str], target) -> np.ndarray:
    y = train_df[target].to_numpy(dtype=float) if isinstance(target, str) else np.asarray(target, dtype=float)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])
    model.fit(train_df[embedding_columns].to_numpy(dtype=float), y)
    return model.predict(test_df[embedding_columns].to_numpy(dtype=float))


def compute_internal_oof_residual_targets(train_df: pd.DataFrame) -> np.ndarray:
    residuals: list[float] = []
    for row_idx, row in train_df.iterrows():
        inner = train_df.drop(index=train_df.index[row_idx]).reset_index(drop=True)
        pred = compute_rpm_knn_predictions(inner, np.asarray([float(row["rpm"])], dtype=float))[0]
        residuals.append(float(row["wind_speed"]) - float(pred))
    return np.asarray(residuals, dtype=float)


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float)
    preds: list[float] = []
    for rpm in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm)
        order = np.argsort(distances)[: min(K_NEIGHBORS, len(train_df))]
        weights = 1.0 / np.maximum(distances[order], EPS)
        weights = weights / weights.sum()
        preds.append(float(np.dot(weights, train_wind[order])))
    return np.asarray(preds, dtype=float)


def build_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, variant_name), block in pred_df.groupby(["protocol", "variant_name"], sort=False):
        true_values = block["true_wind_speed"].to_numpy(dtype=float)
        pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
        signed_error = pred_values - true_values
        rows.append(
            {
                "protocol": protocol,
                "variant_name": variant_name,
                "case_count": int(len(block)),
                "case_mae": float(mean_absolute_error(true_values, pred_values)),
                "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
                "mean_signed_error": float(np.mean(signed_error)),
                "max_abs_error": float(np.max(np.abs(signed_error))),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_knn_neighbors(embedding_case_df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    matrix = StandardScaler().fit_transform(embedding_case_df[embedding_columns].to_numpy(dtype=float))
    case_ids = embedding_case_df["case_id"].to_numpy(dtype=int)
    domains = embedding_case_df["raw_source_domain"].tolist()
    distances = np.sqrt(np.sum(np.square(matrix[:, None, :] - matrix[None, :, :]), axis=2))
    rows: list[dict[str, object]] = []
    for row_idx, case_id in enumerate(case_ids):
        order = np.argsort(distances[row_idx])
        order = order[order != row_idx][: min(K_NEIGHBORS, len(case_ids) - 1)]
        for rank, col_idx in enumerate(order, start=1):
            rows.append(
                {
                    "case_id": int(case_id),
                    "domain": domains[row_idx],
                    "neighbor_rank": rank,
                    "neighbor_case_id": int(case_ids[col_idx]),
                    "neighbor_domain": domains[col_idx],
                    "distance": float(distances[row_idx, col_idx]),
                    "same_domain": bool(domains[row_idx] == domains[col_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_pca_table(embedding_case_df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    matrix = StandardScaler().fit_transform(embedding_case_df[embedding_columns].to_numpy(dtype=float))
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    result = embedding_case_df[["case_id", "file_name", "raw_source_domain", "wind_speed", "rpm", "is_labeled"]].copy()
    result["pca1"] = coords[:, 0]
    result["pca2"] = coords[:, 1]
    return result


def write_summary(output_path: Path, summary_df: pd.DataFrame, pred_df: pd.DataFrame, knn_df: pd.DataFrame) -> None:
    lines = [
        "# contrastive condition embedding quickcheck",
        "",
        "- 状态：`current`",
        "- 首次确认：`2026-04-12`",
        "- 最近复核：`2026-04-12`",
        "- 训练 encoder：`final` 带标签工况 + `added` 带标签工况",
        "- residual 训练：`added(21-24)`",
        "- 测试：`added2(25-30)`",
        "- 风速监督：encoder 训练阶段不使用 `wind_speed`；residual ridge 评估阶段使用 `added` 标签。",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{float(row['case_mae']):.4f}`, "
            f"case_rmse=`{float(row['case_rmse']):.4f}`, max_abs_error=`{float(row['max_abs_error']):.4f}`"
        )
    lines.extend(["", "## Case Predictions", ""])
    for _, row in pred_df.sort_values(["variant_name", "case_id"]).iterrows():
        lines.append(
            f"- `{row['variant_name']}` / `工况{int(row['case_id'])}`: "
            f"pred=`{float(row['pred_wind_speed']):.4f}`, true=`{float(row['true_wind_speed']):.4f}`, "
            f"abs_error=`{float(row['abs_error']):.4f}`"
        )
    lines.extend(["", "## KNN Neighbors", ""])
    for case_id in sorted(knn_df["case_id"].unique()):
        block = knn_df.loc[knn_df["case_id"] == case_id].sort_values("neighbor_rank")
        neighbors = ", ".join(f"{int(row['neighbor_case_id'])}({row['neighbor_domain']}, {float(row['distance']):.2f})" for _, row in block.iterrows())
        lines.append(f"- `工况{int(case_id)}` -> {neighbors}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
