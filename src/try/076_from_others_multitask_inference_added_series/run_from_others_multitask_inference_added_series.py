from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "076_from_others_multitask_inference_added_series"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
MODEL_SCRIPT_PATH = REPO_ROOT / "src" / "from_others" / "2" / "模型训练.py"
MODEL_WEIGHT_PATH = REPO_ROOT / "src" / "from_others" / "2" / "wind_model.pth"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"

WINDOW_SIZE = 100
STEP_SIZE = 50
SENSOR_COLS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 from_others/2 现成权重对 added 系列数据集做推理。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_module = load_module("from_others_multitask_076", MODEL_SCRIPT_PATH)
    device = torch.device("cpu")
    model = model_module.MultiTaskLSTM(input_size=SENSOR_COLS).to(device)
    state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    case_df = build_case_table()
    rows: list[dict[str, object]] = []
    for _, case_row in case_df.iterrows():
        windows = csv_to_windows(Path(case_row["file_path"]))
        pred_wind, pred_rpm = predict_case(model, windows, device=device, batch_size=args.batch_size)
        signed_error = pred_wind - float(case_row["true_wind_speed"])
        rows.append(
            {
                "case_id": int(case_row["case_id"]),
                "file_name": str(case_row["file_name"]),
                "domain": str(case_row["domain"]),
                "true_wind_speed": float(case_row["true_wind_speed"]),
                "true_rpm": float(case_row["rpm"]),
                "pred_wind_speed": float(pred_wind),
                "pred_rpm": float(pred_rpm),
                "signed_error": float(signed_error),
                "abs_error": float(abs(signed_error)),
                "window_count": int(len(windows)),
            }
        )

    case_level_df = pd.DataFrame(rows).sort_values(["domain", "case_id"]).reset_index(drop=True)
    summary_by_domain_df = build_summary_by_domain(case_level_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_domain_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_by_domain_df, case_level_df)

    best_domain = summary_by_domain_df.iloc[0]
    print("076 from_others multitask inference added series 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best domain={best_domain['domain']} | "
        f"case_mae={best_domain['case_mae']:.4f}"
    )


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_case_table() -> pd.DataFrame:
    frames = [
        load_manifest_as_cases(ADDED_MANIFEST_PATH, ADDED_DIR, "added"),
        load_manifest_as_cases(ADDED2_MANIFEST_PATH, ADDED2_DIR, "added2"),
    ]
    return pd.concat(frames, ignore_index=True).sort_values("case_id").reset_index(drop=True)


def load_manifest_as_cases(manifest_path: Path, data_dir: Path, domain: str) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        wind_speed = row.get("wind_speed")
        rpm = row.get("rpm")
        if pd.isna(wind_speed) or pd.isna(rpm):
            continue
        case_id = int(row["case_id"])
        rows.append(
            {
                "case_id": case_id,
                "file_name": f"工况{case_id}.csv",
                "file_path": data_dir / f"工况{case_id}.csv",
                "domain": domain,
                "true_wind_speed": float(wind_speed),
                "rpm": float(rpm),
            }
        )
    return pd.DataFrame(rows)


def csv_to_windows(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if df.shape[1] > SENSOR_COLS:
        df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors="coerce")

    axes = ["AccX", "AccY", "AccZ"]
    for axis in axes:
        col_00005 = f"WSMS00005.{axis}"
        col_00006 = f"WSMS00006.{axis}"
        if col_00006 in df.columns:
            if col_00005 not in df.columns:
                df[col_00005] = np.nan
            df[col_00005] = df[col_00006]

    target_cols: list[str] = []
    for i in range(1, 6):
        sensor_id = f"WSMS0000{i}"
        for axis in axes:
            target_cols.append(f"{sensor_id}.{axis}")
    for i in range(1, 6):
        target_cols.append(f"应变传感器{i}.chdata")

    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[target_cols]
    df = df.fillna(0.0)
    data_array = df.to_numpy(dtype=np.float32)

    windows: list[np.ndarray] = []
    for start in range(0, len(data_array) - WINDOW_SIZE + 1, STEP_SIZE):
        segment = data_array[start : start + WINDOW_SIZE]
        if segment.shape == (WINDOW_SIZE, SENSOR_COLS):
            windows.append(segment.T)
    if not windows:
        raise RuntimeError(f"窗口数为 0: {csv_path}")
    return np.stack(windows, axis=0).astype(np.float32)


def predict_case(model, windows: np.ndarray, *, device: torch.device, batch_size: int) -> tuple[float, float]:
    dataset = TensorDataset(torch.from_numpy(windows).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            pred_labels, _ = model(batch_x)
            preds.append(pred_labels.cpu().numpy())
    pred_array = np.concatenate(preds, axis=0)
    return float(pred_array[:, 0].mean()), float(pred_array[:, 1].mean())


def summarize_block(block: pd.DataFrame) -> dict[str, object]:
    true_values = block["true_wind_speed"].to_numpy(dtype=float)
    pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    return {
        "case_count": int(len(block)),
        "case_mae": float(mean_absolute_error(true_values, pred_values)),
        "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mean_signed_error": float(np.mean(signed_error)),
        "avg_window_count": float(block["window_count"].mean()),
    }


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, block in case_level_df.groupby("domain", sort=False):
        row = {"domain": domain}
        row.update(summarize_block(block))
        rows.append(row)
    rows.append({"domain": "all_external", **summarize_block(case_level_df)})
    return pd.DataFrame(rows).sort_values(["case_mae", "case_rmse", "domain"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, summary_by_domain_df: pd.DataFrame, case_level_df: pd.DataFrame) -> None:
    lines = ["# from_others multitask inference added series", "", "## Summary By Domain", ""]
    for _, row in summary_by_domain_df.iterrows():
        lines.append(
            f"- `{row['domain']}`: case_mae=`{row['case_mae']:.4f}`, "
            f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:+.4f}`, "
            f"avg_window_count=`{row['avg_window_count']:.1f}`"
        )
    lines.extend(["", "## Per Case", ""])
    for domain, block in case_level_df.groupby("domain", sort=False):
        lines.append(f"### {domain}")
        lines.append("")
        for _, row in block.sort_values("case_id").iterrows():
            lines.append(
                f"- `case{int(row['case_id'])}`: pred_wind=`{row['pred_wind_speed']:.4f}`, "
                f"pred_rpm=`{row['pred_rpm']:.4f}`, abs_error=`{row['abs_error']:.4f}`, "
                f"signed_error=`{row['signed_error']:+.4f}`, windows=`{int(row['window_count'])}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
