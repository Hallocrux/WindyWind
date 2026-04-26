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

from tcn_gate_shared import TorchGateConfig, load_classifier_checkpoint, load_gate_tables, normalize_windows, predict_classifier_case, save_checkpoint, stack_windows, train_classifier

TRY_NAME = "050_tcn_conservative_gate_persist"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ENABLE_THRESHOLD = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并持久化 TCN conservative gate。")
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
    dataset_df["enable_enhanced"] = (dataset_df["optimal_gate_target"] > 0.0).astype(int)
    dataset_df["stage2_label"] = dataset_df["optimal_gate_target"].apply(stage2_label)
    config = TorchGateConfig(batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)

    rows: list[dict[str, object]] = []
    for case_id in dataset_df["case_id"]:
        train_case_df = dataset_df.loc[dataset_df["case_id"] != case_id].reset_index(drop=True)
        test_case = dataset_df.loc[dataset_df["case_id"] == case_id].iloc[0]
        train_window_df = window_df.loc[window_df["case_id"] != case_id].reset_index(drop=True)
        test_window_df = window_df.loc[window_df["case_id"] == case_id].reset_index(drop=True)
        train_windows = stack_windows(train_window_df)
        test_windows = stack_windows(test_window_df)

        stage1_targets = train_window_df["case_id"].map(train_case_df.set_index("case_id")["enable_enhanced"]).to_numpy(dtype=np.int64)
        stage1_ckpt = ckpt_dir / f"stage1_case_{case_id}.pt"
        stage1_norm = ckpt_dir / f"stage1_case_{case_id}_norm.npz"
        if stage1_ckpt.exists() and stage1_norm.exists() and not args.force_retrain:
            stage1_model, mean1, std1 = load_classifier_checkpoint(stage1_ckpt, stage1_norm, train_windows.shape[1], config.hidden_channels, 2)
        else:
            train_norm, valid_norm, mean1, std1 = normalize_windows(train_windows, test_windows)
            valid_targets = np.full(len(valid_norm), int(test_case["enable_enhanced"]), dtype=np.int64)
            stage1_model = train_classifier(train_norm, stage1_targets, valid_norm, valid_targets, 2, config)
            save_checkpoint(stage1_model, stage1_ckpt, stage1_norm, mean1, std1, {"case_id": int(case_id), "stage": "enable", "seed": config.seed})
        stage1_probs = predict_classifier_case(stage1_model, test_windows, mean1, std1)
        enable_prob = float(stage1_probs[1]) if len(stage1_probs) > 1 else 0.0

        positive_case_df = train_case_df.loc[train_case_df["enable_enhanced"] == 1].copy()
        positive_window_df = train_window_df.loc[train_window_df["case_id"].isin(positive_case_df["case_id"])].copy()
        if len(positive_window_df) == 0:
            stage2_weight = 0.3
        else:
            pos_windows = stack_windows(positive_window_df)
            stage2_targets = positive_window_df["case_id"].map(positive_case_df.set_index("case_id")["stage2_label"]).to_numpy(dtype=np.int64)
            stage2_ckpt = ckpt_dir / f"stage2_case_{case_id}.pt"
            stage2_norm = ckpt_dir / f"stage2_case_{case_id}_norm.npz"
            if stage2_ckpt.exists() and stage2_norm.exists() and not args.force_retrain:
                stage2_model, mean2, std2 = load_classifier_checkpoint(stage2_ckpt, stage2_norm, pos_windows.shape[1], config.hidden_channels, 3)
            else:
                train_norm2, valid_norm2, mean2, std2 = normalize_windows(pos_windows, test_windows)
                valid_targets2 = np.full(len(valid_norm2), int(stage2_label(float(test_case["optimal_gate_target"]))), dtype=np.int64)
                stage2_model = train_classifier(train_norm2, stage2_targets, valid_norm2, valid_targets2, 3, config)
                save_checkpoint(stage2_model, stage2_ckpt, stage2_norm, mean2, std2, {"case_id": int(case_id), "stage": "bucket", "seed": config.seed})
            stage2_probs = predict_classifier_case(stage2_model, test_windows, mean2, std2)
            stage2_weight = class_to_weight(int(np.argmax(stage2_probs)))

        for variant_name, gate_value in (
            ("tcn_two_stage", stage2_weight if enable_prob >= 0.5 else 0.0),
            (f"tcn_two_stage_t{ENABLE_THRESHOLD:.2f}", stage2_weight if enable_prob >= ENABLE_THRESHOLD else 0.0),
        ):
            pred_wind = (1.0 - gate_value) * float(test_case["pred_base"]) + gate_value * float(test_case["pred_enhanced"])
            rows.append(
                {
                    "variant_name": variant_name,
                    "case_id": int(case_id),
                    "file_name": str(test_case["file_name"]),
                    "domain": str(test_case["domain"]),
                    "true_wind_speed": float(test_case["true_wind_speed"]),
                    "pred_base": float(test_case["pred_base"]),
                    "pred_enhanced": float(test_case["pred_enhanced"]),
                    "pred_gate": float(gate_value),
                    "optimal_gate_target": float(test_case["optimal_gate_target"]),
                    "enable_prob": enable_prob,
                    "pred_wind_speed": float(pred_wind),
                }
            )

    full_windows = stack_windows(window_df)
    enable_targets_full = window_df["case_id"].map(dataset_df.set_index("case_id")["enable_enhanced"]).to_numpy(dtype=np.int64)
    deploy_stage1_ckpt = ckpt_dir / "stage1_deploy_full.pt"
    deploy_stage1_norm = ckpt_dir / "stage1_deploy_full_norm.npz"
    if not (deploy_stage1_ckpt.exists() and deploy_stage1_norm.exists()) or args.force_retrain:
        train_norm, _, mean1, std1 = normalize_windows(full_windows, full_windows)
        stage1_model = train_classifier(train_norm, enable_targets_full, train_norm, enable_targets_full, 2, config)
        save_checkpoint(stage1_model, deploy_stage1_ckpt, deploy_stage1_norm, mean1, std1, {"stage": "enable_deploy", "seed": config.seed})

    positive_window_df_full = window_df.loc[window_df["case_id"].isin(dataset_df.loc[dataset_df["enable_enhanced"] == 1, "case_id"])].copy()
    if len(positive_window_df_full) > 0:
        pos_windows = stack_windows(positive_window_df_full)
        positive_targets_full = positive_window_df_full["case_id"].map(dataset_df.set_index("case_id")["stage2_label"]).to_numpy(dtype=np.int64)
        deploy_stage2_ckpt = ckpt_dir / "stage2_deploy_full.pt"
        deploy_stage2_norm = ckpt_dir / "stage2_deploy_full_norm.npz"
        if not (deploy_stage2_ckpt.exists() and deploy_stage2_norm.exists()) or args.force_retrain:
            train_norm2, _, mean2, std2 = normalize_windows(pos_windows, pos_windows)
            stage2_model = train_classifier(train_norm2, positive_targets_full, train_norm2, positive_targets_full, 3, config)
            save_checkpoint(stage2_model, deploy_stage2_ckpt, deploy_stage2_norm, mean2, std2, {"stage": "bucket_deploy", "seed": config.seed})

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
    print("050 TCN conservative gate 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best all_labeled: {best['variant_name']} | case_mae={best['case_mae']:.4f}")


def stage2_label(value: float) -> int:
    if value >= 0.75:
        return 2
    if value >= 0.4:
        return 1
    return 0


def class_to_weight(class_id: int) -> float:
    return {0: 0.3, 1: 0.5, 2: 1.0}[int(class_id)]


def summarize(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, block in [("all_labeled", case_df), *case_df.groupby("domain", sort=False)]:
        for variant_name, variant_block in block.groupby("variant_name", sort=False):
            rows.append(
                {
                    "domain": domain_name,
                    "variant_name": variant_name,
                    "case_mae": float(variant_block["abs_error"].mean()),
                    "case_rmse": float(np.sqrt(np.mean(np.square(variant_block["signed_error"])))),
                    "mean_signed_error": float(variant_block["signed_error"].mean()),
                    "mean_gate": float(variant_block["pred_gate"].mean()),
                    "gate_std": float(variant_block["pred_gate"].std(ddof=0)),
                    "case_count": int(len(variant_block)),
                }
            )
    return pd.DataFrame(rows)


def write_summary(output_path: Path, summary_df: pd.DataFrame) -> None:
    lines = [f"# {TRY_NAME}", "", "## Summary", ""]
    for _, row in summary_df.iterrows():
        lines.append(f"- `{row['domain']} | {row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, mean_gate=`{row['mean_gate']:.4f}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
