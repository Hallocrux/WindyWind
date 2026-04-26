from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY013_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase3_cnn_tcn_lib import TinyTCN, TorchTrainConfig, train_torch_model  # noqa: E402
from phase3_end_to_end_lib import build_raw_window_dataset  # noqa: E402
from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records  # noqa: E402
from src.current.features import WindowConfig  # noqa: E402

TRY_NAME = "063_final_late_fusion_added2_replay"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DATA_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
WINDOW_CONFIGS = {
    "2s": WindowConfig(sampling_rate=50.0, window_size=100, step_size=50),
    "8s": WindowConfig(sampling_rate=50.0, window_size=400, step_size=200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="final 训练的 2s+8s 晚融合回放 added / added2。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_manifest_records(ADDED_MANIFEST_PATH, ADDED_DATA_DIR)
    added2_records = load_manifest_records(ADDED2_MANIFEST_PATH, ADDED2_DATA_DIR)
    all_records = [*final_records, *added_records, *added2_records]
    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    prediction_frames: dict[str, pd.DataFrame] = {}
    for order, (window_label, window_config) in enumerate(WINDOW_CONFIGS.items(), start=1):
        prediction_frames[window_label] = train_or_load_and_predict(
            final_records=final_records,
            eval_records=[*added_records, *added2_records],
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            window_label=window_label,
            train_config=train_config,
            checkpoint_dir=ckpt_dir,
            seed=args.random_seed + order * 1000,
            force_retrain=args.force_retrain,
        )

    case_level_df = merge_predictions(prediction_frames, added_records, added2_records)
    summary_df = build_summary_by_domain(case_level_df)
    write_summary_markdown(output_dir / "summary.md", summary_df, case_level_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")

    best_added2 = summary_df.loc[summary_df["domain"] == "added2"].iloc[0]
    print("063 final 训练的 2s+8s 晚融合回放已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best added2: {best_added2['model_name']} | case_mae={best_added2['case_mae']:.4f}")


def load_manifest_records(manifest_path: Path, data_dir: Path) -> list[DatasetRecord]:
    manifest_df = pd.read_csv(manifest_path)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        wind_speed = pd.to_numeric(pd.Series([row["wind_speed"]]), errors="coerce").iloc[0]
        rpm = pd.to_numeric(pd.Series([row["rpm"]]), errors="coerce").iloc[0]
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=data_dir / f"工况{case_id}.csv",
                wind_speed=float(wind_speed) if pd.notna(wind_speed) else None,
                rpm=float(rpm) if pd.notna(rpm) else None,
                is_labeled=pd.notna(wind_speed) and pd.notna(rpm),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return [record for record in records if record.is_labeled]


def train_or_load_and_predict(
    *,
    final_records: list[DatasetRecord],
    eval_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    window_label: str,
    train_config: TorchTrainConfig,
    checkpoint_dir: Path,
    seed: int,
    force_retrain: bool,
) -> pd.DataFrame:
    ckpt_path = checkpoint_dir / f"final_deploy_{window_label}.pt"
    norm_path = checkpoint_dir / f"final_deploy_{window_label}_norm.npz"
    meta_path = checkpoint_dir / f"final_deploy_{window_label}.json"

    train_dataset = build_raw_window_dataset(
        final_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in final_records},
        window_config,
    )
    eval_dataset = build_raw_window_dataset(
        eval_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in eval_records},
        window_config,
    )
    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows

    X_train_norm, X_eval_norm, mean, std = normalize_windows_by_channel(X_train, X_eval)
    model = TinyTCN(in_channels=X_train.shape[1])

    if ckpt_path.exists() and norm_path.exists() and meta_path.exists() and not force_retrain:
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
        model = model.to(device)
        train_torch_model(
            model=model,
            X_train=X_train_norm,
            y_train=y_train,
            X_valid=X_train_norm,
            y_valid=y_train,
            config=train_config,
            device=device,
        )
        torch.save(model.state_dict(), ckpt_path)
        np.savez(norm_path, mean=mean, std=std)
        meta_path.write_text(
            json.dumps(
                {
                    "window_label": window_label,
                    "seed": seed,
                    "train_case_ids": [record.case_id for record in final_records],
                    "train_case_count": len(final_records),
                    "window_size": window_config.window_size,
                    "step_size": window_config.step_size,
                    "max_epochs": train_config.max_epochs,
                    "patience": train_config.patience,
                    "learning_rate": train_config.learning_rate,
                    "weight_decay": train_config.weight_decay,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = model.cpu()

    with torch.no_grad():
        pred = model(torch.from_numpy(X_eval_norm).float()).cpu().numpy()

    pred_df = eval_dataset.meta_df[["case_id", "file_name", "wind_speed", "rpm"]].copy()
    pred_df["pred_wind_speed"] = pred.astype(float)
    pred_df["window_label"] = window_label
    return (
        pred_df.groupby(["case_id", "file_name", "wind_speed", "rpm", "window_label"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"wind_speed": "true_wind_speed"})
    )


def normalize_windows_by_channel(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    X_train_norm = ((X_train - channel_mean) / channel_std).astype(np.float32)
    X_eval_norm = ((X_eval - channel_mean) / channel_std).astype(np.float32)
    return X_train_norm, X_eval_norm, channel_mean.astype(np.float32), channel_std.astype(np.float32)


def merge_predictions(
    prediction_frames: dict[str, pd.DataFrame],
    added_records: list[DatasetRecord],
    added2_records: list[DatasetRecord],
) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for window_label, frame in prediction_frames.items():
        block = frame.rename(columns={"pred_wind_speed": f"pred_{window_label}"})
        if base is None:
            base = block[["case_id", "file_name", "true_wind_speed", "rpm", f"pred_{window_label}"]].copy()
        else:
            base = base.merge(block[["case_id", f"pred_{window_label}"]], on="case_id", how="left")
    assert base is not None

    added_ids = {record.case_id for record in added_records}
    added2_ids = {record.case_id for record in added2_records}
    base["domain"] = base["case_id"].map(
        lambda case_id: "added" if int(case_id) in added_ids else ("added2" if int(case_id) in added2_ids else "unknown")
    )
    base["pred_2s_8s_fusion"] = 0.5 * (base["pred_2s"] + base["pred_8s"])

    for model_name in ("2s", "8s", "2s_8s_fusion"):
        pred_col = f"pred_{model_name}"
        signed_col = f"signed_error_{model_name}"
        abs_col = f"abs_error_{model_name}"
        base[signed_col] = base[pred_col] - base["true_wind_speed"]
        base[abs_col] = base[signed_col].abs()
    return base.sort_values(["domain", "case_id"]).reset_index(drop=True)


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name in ("added", "added2", "all_external"):
        subset = case_level_df if domain_name == "all_external" else case_level_df.loc[case_level_df["domain"] == domain_name].copy()
        for model_name in ("2s", "8s", "2s_8s_fusion"):
            signed_error = subset[f"signed_error_{model_name}"].to_numpy(dtype=float, copy=False)
            abs_error = subset[f"abs_error_{model_name}"].to_numpy(dtype=float, copy=False)
            rows.append(
                {
                    "domain": domain_name,
                    "model_name": model_name,
                    "case_mae": float(np.mean(abs_error)),
                    "case_rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
                    "mean_signed_error": float(np.mean(signed_error)),
                    "case_count": int(len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "model_name"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, summary_df: pd.DataFrame, case_level_df: pd.DataFrame) -> None:
    lines = [
        "# final 训练的 2s+8s 晚融合回放",
        "",
        f"- `added` 工况数：`{int((case_level_df['domain'] == 'added').sum())}`",
        f"- `added2` 工况数：`{int((case_level_df['domain'] == 'added2').sum())}`",
        "",
        "## 外部域汇总",
        "",
    ]
    for domain_name in ("added", "added2", "all_external"):
        lines.append(f"### {domain_name}")
        lines.append("")
        block = summary_df.loc[summary_df["domain"] == domain_name]
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['model_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`"
            )
        lines.append("")

    lines.append("## 每工况预测")
    lines.append("")
    for _, row in case_level_df.iterrows():
        lines.append(
            f"- `{row['domain']} | 工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, "
            f"pred_2s=`{row['pred_2s']:.4f}`, pred_8s=`{row['pred_8s']:.4f}`, "
            f"pred_fusion=`{row['pred_2s_8s_fusion']:.4f}`, abs_fusion=`{row['abs_error_2s_8s_fusion']:.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
