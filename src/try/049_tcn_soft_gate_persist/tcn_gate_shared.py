from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

TRY047_SCRIPT = REPO_ROOT / "src" / "try" / "047_soft_gate_quickcheck" / "run_soft_gate_quickcheck.py"
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)


@dataclass(frozen=True)
class TorchGateConfig:
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: int = 32
    seed: int = 42


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


class TinyTCNGateRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            TemporalBlock(in_channels, hidden_channels, 1),
            TemporalBlock(hidden_channels, hidden_channels, 2),
            TemporalBlock(hidden_channels, hidden_channels, 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class TinyTCNGateClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            TemporalBlock(in_channels, hidden_channels, 1),
            TemporalBlock(hidden_channels, hidden_channels, 2),
            TemporalBlock(hidden_channels, hidden_channels, 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_try047_module():
    spec = importlib.util.spec_from_file_location("try047_tcn_gate", TRY047_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {TRY047_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["try047_tcn_gate"] = module
    spec.loader.exec_module(module)
    return module


def load_gate_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    try047 = load_try047_module()
    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = try047.load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)
    dataset_df = try047.build_gate_feature_table(all_records, common_signal_columns).merge(
        try047.build_expert_prediction_table(),
        on=["case_id", "file_name", "true_wind_speed", "domain"],
        how="inner",
    )
    dataset_df["pred_gap"] = dataset_df["pred_enhanced"] - dataset_df["pred_base"]
    dataset_df["abs_pred_gap"] = dataset_df["pred_gap"].abs()
    dataset_df["optimal_gate_target"] = try047.compute_optimal_gate_target(
        true_values=dataset_df["true_wind_speed"].to_numpy(dtype=float),
        pred_base=dataset_df["pred_base"].to_numpy(dtype=float),
        pred_enhanced=dataset_df["pred_enhanced"].to_numpy(dtype=float),
    )

    rows: list[dict[str, object]] = []
    target_lookup = dataset_df.set_index("case_id")["optimal_gate_target"].to_dict()
    pred_base_lookup = dataset_df.set_index("case_id")["pred_base"].to_dict()
    pred_enhanced_lookup = dataset_df.set_index("case_id")["pred_enhanced"].to_dict()
    for record in all_records:
        cleaned_df = load_clean_signal_frame(record, common_signal_columns)
        for _, segment_df in cleaned_df.groupby("__segment_id", sort=True):
            total_rows = len(segment_df)
            if total_rows < WINDOW_CONFIG.window_size:
                continue
            segment_df = segment_df.reset_index(drop=True)
            for start in range(0, total_rows - WINDOW_CONFIG.window_size + 1, WINDOW_CONFIG.step_size):
                window = segment_df.iloc[start : start + WINDOW_CONFIG.window_size]
                rows.append(
                    {
                        "case_id": record.case_id,
                        "file_name": record.file_name,
                        "domain": "added" if record.case_id >= 21 else "final",
                        "true_wind_speed": float(record.wind_speed),
                        "pred_base": float(pred_base_lookup[record.case_id]),
                        "pred_enhanced": float(pred_enhanced_lookup[record.case_id]),
                        "gate_target": float(target_lookup[record.case_id]),
                        "window_array": window[common_signal_columns].to_numpy(dtype=float, copy=False).T.copy(),
                    }
                )
    return dataset_df.sort_values("case_id").reset_index(drop=True), pd.DataFrame(rows), common_signal_columns


def stack_windows(window_df: pd.DataFrame) -> np.ndarray:
    return np.stack(window_df["window_array"].to_list(), axis=0)


def normalize_windows(train_windows: np.ndarray, eval_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_windows.mean(axis=(0, 2), keepdims=True)
    std = train_windows.std(axis=(0, 2), keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return ((train_windows - mean) / std).astype(np.float32), ((eval_windows - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_regressor(train_windows: np.ndarray, train_targets: np.ndarray, valid_windows: np.ndarray, valid_targets: np.ndarray, config: TorchGateConfig) -> TinyTCNGateRegressor:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = TinyTCNGateRegressor(in_channels=train_windows.shape[1], hidden_channels=config.hidden_channels)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_targets).float()), batch_size=config.batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    valid_x = torch.from_numpy(valid_windows).float()
    valid_y = torch.from_numpy(valid_targets).float()
    best_state = None
    best_loss = float("inf")
    patience_left = config.patience
    for _ in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(criterion(model(valid_x), valid_y).item())
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
    return model


def train_classifier(train_windows: np.ndarray, train_targets: np.ndarray, valid_windows: np.ndarray, valid_targets: np.ndarray, num_classes: int, config: TorchGateConfig) -> TinyTCNGateClassifier:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = TinyTCNGateClassifier(in_channels=train_windows.shape[1], num_classes=num_classes, hidden_channels=config.hidden_channels)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_targets).long()), batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    valid_x = torch.from_numpy(valid_windows).float()
    valid_y = torch.from_numpy(valid_targets).long()
    best_state = None
    best_loss = float("inf")
    patience_left = config.patience
    for _ in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(criterion(model(valid_x), valid_y).item())
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
    return model


def save_checkpoint(model: nn.Module, ckpt_path: Path, norm_path: Path, mean: np.ndarray, std: np.ndarray, metadata: dict[str, object]) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    np.savez(norm_path, mean=mean, std=std)
    ckpt_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_regressor_checkpoint(ckpt_path: Path, norm_path: Path, in_channels: int, hidden_channels: int) -> tuple[TinyTCNGateRegressor, np.ndarray, np.ndarray]:
    model = TinyTCNGateRegressor(in_channels=in_channels, hidden_channels=hidden_channels)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    norm = np.load(norm_path)
    return model.eval(), norm["mean"], norm["std"]


def load_classifier_checkpoint(ckpt_path: Path, norm_path: Path, in_channels: int, hidden_channels: int, num_classes: int) -> tuple[TinyTCNGateClassifier, np.ndarray, np.ndarray]:
    model = TinyTCNGateClassifier(in_channels=in_channels, num_classes=num_classes, hidden_channels=hidden_channels)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    norm = np.load(norm_path)
    return model.eval(), norm["mean"], norm["std"]


def predict_regressor_case(model: TinyTCNGateRegressor, eval_windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    eval_norm = ((eval_windows - mean) / std).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(eval_norm).float()).cpu().numpy()
    return float(np.clip(np.mean(pred), 0.0, 1.0))


def predict_classifier_case(model: TinyTCNGateClassifier, eval_windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    eval_norm = ((eval_windows - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(eval_norm).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.mean(axis=0)
