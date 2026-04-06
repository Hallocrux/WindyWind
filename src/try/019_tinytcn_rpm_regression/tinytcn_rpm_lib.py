from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from phase3_end_to_end_lib import RawDataset
else:
    RawDataset = Any


@dataclass(frozen=True)
class TorchTrainConfig:
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 8
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


def evaluate_tinytcn_rpm_loco(
    dataset: RawDataset,
    train_config: TorchTrainConfig,
    random_seed: int = 42,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["rpm"].notna().to_numpy()
    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = dataset.windows[labeled_mask]
    split_map = _build_split_map_from_meta(labeled_meta)
    y_all = labeled_meta["rpm"].to_numpy(dtype=np.float32, copy=False)
    predictions: list[pd.DataFrame] = []

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cpu")

    for _, (train_idx, valid_idx) in split_map.items():
        X_train = labeled_windows[train_idx]
        X_valid = labeled_windows[valid_idx]
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]

        X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)
        model = TinyTCN(in_channels=X_train.shape[1]).to(device)
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
            pred = model(torch.from_numpy(X_valid_norm).to(device)).cpu().numpy()

        valid_df = labeled_meta.iloc[valid_idx][
            ["case_id", "file_name", "window_index", "start_time", "end_time", "rpm"]
        ].copy()
        valid_df = valid_df.rename(columns={"rpm": "true_rpm"})
        valid_df["pred_rpm"] = pred
        predictions.append(valid_df)

    return pd.concat(predictions, ignore_index=True)


def predict_tinytcn_rpm_unlabeled(
    dataset: RawDataset,
    train_config: TorchTrainConfig,
    random_seed: int = 42,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["rpm"].notna().to_numpy()
    unlabeled_mask = dataset.meta_df["rpm"].isna().to_numpy()
    if not unlabeled_mask.any():
        return pd.DataFrame(columns=["case_id", "file_name", "predicted_rpm"])

    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    unlabeled_meta = dataset.meta_df.loc[unlabeled_mask].reset_index(drop=True)
    X_train = dataset.windows[labeled_mask]
    X_unlabeled = dataset.windows[unlabeled_mask]
    y_train = labeled_meta["rpm"].to_numpy(dtype=np.float32, copy=False)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cpu")
    X_train_norm, X_unlabeled_norm = normalize_windows_by_channel(X_train, X_unlabeled)
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
        pred = model(torch.from_numpy(X_unlabeled_norm).to(device)).cpu().numpy()
    pred_df = unlabeled_meta[["case_id", "file_name"]].copy()
    pred_df["pred_rpm"] = pred
    return (
        pred_df.groupby(["case_id", "file_name"], as_index=False)["pred_rpm"]
        .mean()
        .rename(columns={"pred_rpm": "predicted_rpm"})
    )


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: TorchTrainConfig,
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
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def normalize_windows_by_channel(
    X_train: np.ndarray,
    X_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    return (
        ((X_train - channel_mean) / channel_std).astype(np.float32),
        ((X_valid - channel_mean) / channel_std).astype(np.float32),
    )


def summarize_rpm_predictions(
    prediction_frame: pd.DataFrame,
    model_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    case_df = (
        prediction_frame.groupby(["case_id", "file_name", "true_rpm"], as_index=False)["pred_rpm"]
        .mean()
        .rename(columns={"pred_rpm": "pred_mean"})
    )
    case_df["abs_error"] = (case_df["pred_mean"] - case_df["true_rpm"]).abs()
    case_df["model_name"] = model_name
    errors = case_df["pred_mean"] - case_df["true_rpm"]
    summary_row = {
        "model_name": model_name,
        "case_mae": float(np.mean(np.abs(errors))),
        "case_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "case_mape": float(
            np.mean(np.abs(errors) / case_df["true_rpm"].to_numpy(dtype=float)) * 100
        ),
    }
    return summary_row, case_df


def _build_split_map_from_meta(meta_df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    case_ids = sorted(meta_df["case_id"].unique())
    case_values = meta_df["case_id"].to_numpy(dtype=int, copy=False)
    return {
        case_id: (
            np.flatnonzero(case_values != case_id),
            np.flatnonzero(case_values == case_id),
        )
        for case_id in case_ids
    }

