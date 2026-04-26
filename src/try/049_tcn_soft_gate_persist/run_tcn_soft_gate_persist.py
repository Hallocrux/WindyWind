from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SOFT_DIR = REPO_ROOT / "src" / "try" / "049_tcn_soft_gate_persist"
for path in (REPO_ROOT, SOFT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tcn_gate_shared import TorchGateConfig, load_gate_tables, load_regressor_checkpoint, normalize_windows, predict_regressor_case, save_checkpoint, stack_windows, train_regressor

TRY_NAME = "049_tcn_soft_gate_persist"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并持久化 TCN soft gate。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "models" / "checkpoints"
    dataset_df, window_df, _ = load_gate_tables()
    config = TorchGateConfig(batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)

    rows: list[dict[str, object]] = []
    for case_id in dataset_df["case_id"]:
        train_window_df = window_df.loc[window_df["case_id"] != case_id].reset_index(drop=True)
        test_window_df = window_df.loc[window_df["case_id"] == case_id].reset_index(drop=True)
        train_windows = stack_windows(train_window_df)
        test_windows = stack_windows(test_window_df)
        train_targets = train_window_df["gate_target"].to_numpy(dtype=np.float32)
        test_case = dataset_df.loc[dataset_df["case_id"] == case_id].iloc[0]

        ckpt_path = ckpt_dir / f"fold_case_{case_id}.pt"
        norm_path = ckpt_dir / f"fold_case_{case_id}_norm.npz"
        if ckpt_path.exists() and norm_path.exists() and not args.force_retrain:
            model, mean, std = load_regressor_checkpoint(ckpt_path, norm_path, train_windows.shape[1], config.hidden_channels)
        else:
            train_norm, valid_norm, mean, std = normalize_windows(train_windows, test_windows)
            valid_targets = np.full(len(valid_norm), float(test_case["optimal_gate_target"]), dtype=np.float32)
            model = train_regressor(train_norm, train_targets, valid_norm, valid_targets, config)
            save_checkpoint(model, ckpt_path, norm_path, mean, std, {"case_id": int(case_id), "scheme": "soft", "seed": config.seed})
        pred_gate = predict_regressor_case(model, test_windows, mean, std)
        pred_wind = (1.0 - pred_gate) * float(test_case["pred_base"]) + pred_gate * float(test_case["pred_enhanced"])
        rows.append(
            {
                "variant_name": "tcn_soft_gate",
                "case_id": int(case_id),
                "file_name": str(test_case["file_name"]),
                "domain": str(test_case["domain"]),
                "true_wind_speed": float(test_case["true_wind_speed"]),
                "pred_base": float(test_case["pred_base"]),
                "pred_enhanced": float(test_case["pred_enhanced"]),
                "pred_gate": pred_gate,
                "optimal_gate_target": float(test_case["optimal_gate_target"]),
                "pred_wind_speed": float(pred_wind),
            }
        )

    full_windows = stack_windows(window_df)
    full_targets = window_df["gate_target"].to_numpy(dtype=np.float32)
    deploy_ckpt = ckpt_dir / "deploy_full.pt"
    deploy_norm = ckpt_dir / "deploy_full_norm.npz"
    if not (deploy_ckpt.exists() and deploy_norm.exists()) or args.force_retrain:
        train_norm, _, mean, std = normalize_windows(full_windows, full_windows)
        model = train_regressor(train_norm, full_targets, train_norm, full_targets, config)
        save_checkpoint(model, deploy_ckpt, deploy_norm, mean, std, {"scheme": "soft_deploy", "seed": config.seed})

    case_df = pd.DataFrame(rows)
    case_df["signed_error"] = case_df["pred_wind_speed"] - case_df["true_wind_speed"]
    case_df["abs_error"] = case_df["signed_error"].abs()
    summary_df = summarize(case_df)
    write_summary(output_dir / "summary.md", summary_df)
    dataset_df.to_csv(output_dir / "dataset_table.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_variant.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    best = summary_df.loc[summary_df["domain"] == "all_labeled"].iloc[0]
    print("049 TCN soft gate 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best all_labeled: case_mae={best['case_mae']:.4f}")


def summarize(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, block in [("all_labeled", case_df), *case_df.groupby("domain", sort=False)]:
        rows.append(
            {
                "domain": domain_name,
                "variant_name": "tcn_soft_gate",
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "mean_gate": float(block["pred_gate"].mean()),
                "gate_std": float(block["pred_gate"].std(ddof=0)),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows)


def write_summary(output_path: Path, summary_df: pd.DataFrame) -> None:
    lines = [f"# {TRY_NAME}", "", "## Summary", ""]
    for _, row in summary_df.iterrows():
        lines.append(f"- `{row['domain']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, mean_gate=`{row['mean_gate']:.4f}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
