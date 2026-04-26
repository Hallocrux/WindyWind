from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY041_ROOT = REPO_ROOT / "src" / "try" / "041_rpm_vs_learned_midband_check"
TRY042_ROOT = REPO_ROOT / "src" / "try" / "042_rpm_learned_midband_multiseed_stability_check"
for path in (REPO_ROOT, TRY041_ROOT, TRY042_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_rpm_learned_midband_multiseed_stability_check as try042  # noqa: E402
import run_rpm_vs_learned_midband_check as try041  # noqa: E402
from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records  # noqa: E402

TRY_NAME = "064_added_sota_added2_replay"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DATA_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
TRY063_CASE_PATH = REPO_ROOT / "outputs" / "try" / "063_final_late_fusion_added2_replay" / "case_level_predictions.csv"
DEFAULT_BASE_SEEDS = try042.DEFAULT_BASE_SEEDS
TINYTCN_SEED_OFFSET = try042.TINYTCN_SEED_OFFSET
LEARNED_VARIANT_NAME = try042.LEARNED_VARIANT_NAME
TARGET_VARIANT_NAMES = try042.TARGET_VARIANT_NAMES
FUSION_WEIGHTS = (0.3, 0.5, 0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放 added SOTA 与 2s+8s 到 added2。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_BASE_SEEDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.use_deterministic_algorithms(True)

    train_config = try041.TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = try041.load_added_records()
    added2_records = load_manifest_records(ADDED2_MANIFEST_PATH, ADDED2_DATA_DIR)
    eval_records = [*added_records, *added2_records]
    domain_by_case_id = {
        **{record.case_id: "added" for record in added_records},
        **{record.case_id: "added2" for record in added2_records},
    }

    all_records = [*final_records, *eval_records]
    common_signal_columns = get_common_signal_columns(all_records)
    strain_columns = [column for column in common_signal_columns if "应变" in column]
    acc_columns = [column for column in common_signal_columns if "Acc" in column]
    selected_columns = [*strain_columns, *acc_columns]

    base_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }
    midband_frames = try041.build_midband_frames(base_frames, all_records, strain_columns)

    train_dataset = try041.build_selected_window_dataset(final_records, midband_frames, selected_columns)
    eval_dataset = try041.build_selected_window_dataset(eval_records, midband_frames, selected_columns)
    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows
    train_mean, train_std = compute_channel_norm(X_train)

    rpm_df = try041.build_rpm_knn_predictions(eval_records)
    rpm_df["domain"] = rpm_df["case_id"].map(domain_by_case_id)

    seed_case_frames: list[pd.DataFrame] = []
    asset_rows: list[dict[str, object]] = []
    for seed_order, base_seed in enumerate(args.seeds, start=1):
        tinytcn_seed = base_seed + TINYTCN_SEED_OFFSET
        learned_df, asset_row = train_or_load_midband_predictions(
            checkpoint_dir=checkpoint_dir,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            train_mean=train_mean,
            train_std=train_std,
            selected_columns=selected_columns,
            train_records=final_records,
            train_config=train_config,
            base_seed=base_seed,
            seed_order=seed_order,
            tinytcn_seed=tinytcn_seed,
            force_retrain=args.force_retrain,
        )
        learned_df["domain"] = learned_df["case_id"].map(domain_by_case_id)
        seed_case_frames.append(build_seed_case_predictions(rpm_df, learned_df, seed_order, base_seed, tinytcn_seed))
        asset_rows.append(asset_row)

    seed_case_level_df = pd.concat(seed_case_frames, ignore_index=True)
    seed_summary_df = build_seed_summary_by_domain(seed_case_level_df)
    stability_df = build_stability_overview(seed_summary_df)
    direct_summary_df, direct_added2_case_df = load_direct_2s_8s_reference()
    comparison_df = build_comparison_overview(stability_df, direct_summary_df)
    added2_case_comparison_df = build_added2_case_comparison(seed_case_level_df, direct_added2_case_df)
    variant_config_df = try042.build_variant_config_table()
    model_assets_df = pd.DataFrame(asset_rows)

    variant_config_df.to_csv(output_dir / "variant_config_table.csv", index=False, encoding="utf-8-sig")
    seed_case_level_df.to_csv(output_dir / "seed_case_level_predictions.csv", index=False, encoding="utf-8-sig")
    seed_summary_df.to_csv(output_dir / "seed_summary_by_domain.csv", index=False, encoding="utf-8-sig")
    stability_df.to_csv(output_dir / "stability_overview_by_domain.csv", index=False, encoding="utf-8-sig")
    direct_summary_df.to_csv(output_dir / "direct_2s_8s_summary_by_domain.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(output_dir / "comparison_overview.csv", index=False, encoding="utf-8-sig")
    added2_case_comparison_df.to_csv(output_dir / "added2_case_comparison.csv", index=False, encoding="utf-8-sig")
    model_assets_df.to_csv(output_dir / "model_assets.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", stability_df, direct_summary_df, comparison_df, added2_case_comparison_df)

    best_added2 = comparison_df.loc[comparison_df["domain"] == "added2"].iloc[0]
    print("064 added SOTA 回放 added2 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best added2: {best_added2['variant_name']} | case_mae={best_added2['case_mae_mean']:.4f}")


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


def compute_channel_norm(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    return channel_mean.astype(np.float32), channel_std.astype(np.float32)


def train_or_load_midband_predictions(
    *,
    checkpoint_dir: Path,
    train_dataset,
    eval_dataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    selected_columns: list[str],
    train_records: list[DatasetRecord],
    train_config: try041.TrainConfig,
    base_seed: int,
    seed_order: int,
    tinytcn_seed: int,
    force_retrain: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    stem = f"final_deploy_midband_seed{base_seed}_w5s"
    ckpt_path = checkpoint_dir / f"{stem}.pt"
    norm_path = checkpoint_dir / f"{stem}_norm.npz"
    meta_path = checkpoint_dir / f"{stem}.json"

    model = try041.TinyTCN(in_channels=X_train.shape[1])
    if ckpt_path.exists() and norm_path.exists() and meta_path.exists() and not force_retrain:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        norm = np.load(norm_path)
        mean = norm["mean"]
        std = norm["std"]
        action = "loaded"
    else:
        mean = train_mean
        std = train_std
        X_train_norm = ((X_train - mean) / std).astype(np.float32)
        torch.manual_seed(tinytcn_seed)
        np.random.seed(tinytcn_seed)
        device = torch.device("cpu")
        model = model.to(device)
        try041.train_model(model, X_train_norm, y_train, X_train_norm, y_train, train_config, device)
        torch.save(model.cpu().state_dict(), ckpt_path)
        np.savez(norm_path, mean=mean, std=std)
        meta_path.write_text(
            json.dumps(
                {
                    "variant_name": LEARNED_VARIANT_NAME,
                    "base_seed": base_seed,
                    "seed_order": seed_order,
                    "tinytcn_seed": tinytcn_seed,
                    "train_case_ids": [record.case_id for record in train_records],
                    "train_case_count": len(train_records),
                    "window_label": "5s",
                    "window_size": try041.WINDOW_CONFIG.window_size,
                    "step_size": try041.WINDOW_CONFIG.step_size,
                    "input_columns": "all_channels_midband",
                    "strain_transform": "strain_bandpass_3.0_6.0Hz",
                    "selected_columns": selected_columns,
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
        action = "trained"

    X_eval_norm = ((X_eval - mean) / std).astype(np.float32)
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_eval_norm).float()).cpu().numpy()

    pred_df = try041.aggregate_case_predictions(eval_dataset.meta_df, pred)
    pred_df["variant_name"] = LEARNED_VARIANT_NAME
    pred_df["model_family"] = "tinytcn"
    pred_df["input_columns"] = "all_channels_midband"
    pred_df["strain_transform"] = "strain_bandpass_3.0_6.0Hz"
    pred_df["fusion_weight_learned"] = np.nan

    return pred_df, {
        "base_seed": base_seed,
        "tinytcn_seed": tinytcn_seed,
        "action": action,
        "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
        "norm_path": str(norm_path.relative_to(REPO_ROOT)),
        "meta_path": str(meta_path.relative_to(REPO_ROOT)),
    }


def build_seed_case_predictions(
    rpm_df: pd.DataFrame,
    learned_df: pd.DataFrame,
    seed_order: int,
    base_seed: int,
    tinytcn_seed: int,
) -> pd.DataFrame:
    base_columns = ["case_id", "file_name", "true_wind_speed", "rpm", "domain"]
    rpm_block = rpm_df.copy()
    learned_block = learned_df.copy()
    rows = [rpm_block, learned_block]

    merged = rpm_block[base_columns + ["pred_wind_speed"]].rename(columns={"pred_wind_speed": "pred_rpm"}).merge(
        learned_block[base_columns + ["pred_wind_speed"]].rename(columns={"pred_wind_speed": "pred_learned"}),
        on=base_columns,
        how="inner",
    )
    for learned_weight in FUSION_WEIGHTS:
        block = merged[base_columns].copy()
        block["pred_wind_speed"] = (1.0 - learned_weight) * merged["pred_rpm"] + learned_weight * merged["pred_learned"]
        block["variant_name"] = f"fusion_rpm_knn4__tinytcn_all_channels_midband__w{learned_weight:.1f}"
        block["model_family"] = "fusion"
        block["input_columns"] = "rpm + all_channels_midband"
        block["strain_transform"] = "strain_bandpass_3.0_6.0Hz"
        block["fusion_weight_learned"] = learned_weight
        rows.append(block)

    case_level_df = pd.concat(rows, ignore_index=True)
    case_level_df = case_level_df.loc[case_level_df["variant_name"].isin(TARGET_VARIANT_NAMES)].copy()
    case_level_df["seed_order"] = seed_order
    case_level_df["base_seed"] = base_seed
    case_level_df["tinytcn_seed"] = tinytcn_seed
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()
    return case_level_df.sort_values(["domain", "seed_order", "variant_name", "case_id"]).reset_index(drop=True)


def build_seed_summary_by_domain(seed_case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    domain_blocks = [
        ("added", seed_case_level_df.loc[seed_case_level_df["domain"] == "added"].copy()),
        ("added2", seed_case_level_df.loc[seed_case_level_df["domain"] == "added2"].copy()),
        ("all_external", seed_case_level_df.copy()),
    ]
    for domain_name, domain_df in domain_blocks:
        for (seed_order, base_seed, tinytcn_seed, variant_name), block in domain_df.groupby(
            ["seed_order", "base_seed", "tinytcn_seed", "variant_name"],
            sort=False,
        ):
            rows.append(
                {
                    "domain": domain_name,
                    "seed_order": int(seed_order),
                    "base_seed": int(base_seed),
                    "tinytcn_seed": int(tinytcn_seed),
                    "variant_name": variant_name,
                    "model_family": block["model_family"].iloc[0],
                    "input_columns": block["input_columns"].iloc[0],
                    "strain_transform": block["strain_transform"].iloc[0],
                    "fusion_weight_learned": block["fusion_weight_learned"].iloc[0],
                    "case_mae": float(block["abs_error"].mean()),
                    "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                    "mean_signed_error": float(block["signed_error"].mean()),
                    "case_count": int(len(block)),
                }
            )
    result = pd.DataFrame(rows)
    ranked_blocks: list[pd.DataFrame] = []
    for (domain_name, base_seed), block in result.groupby(["domain", "base_seed"], sort=True):
        ordered = block.sort_values(["case_mae", "case_rmse", "variant_name"]).reset_index(drop=True).copy()
        ordered["case_mae_rank"] = np.arange(1, len(ordered) + 1)
        ranked_blocks.append(ordered)
    return pd.concat(ranked_blocks, ignore_index=True).sort_values(["domain", "base_seed", "case_mae_rank"]).reset_index(drop=True)


def build_stability_overview(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain_name, variant_name), block in seed_summary_df.groupby(["domain", "variant_name"], sort=False):
        rows.append(
            {
                "domain": domain_name,
                "variant_name": variant_name,
                "model_family": block["model_family"].iloc[0],
                "input_columns": block["input_columns"].iloc[0],
                "strain_transform": block["strain_transform"].iloc[0],
                "fusion_weight_learned": block["fusion_weight_learned"].iloc[0],
                "seed_count": int(len(block)),
                "best_seed_count": int((block["case_mae_rank"] == 1).sum()),
                "best_seed_rate": float((block["case_mae_rank"] == 1).mean()),
                "case_mae_mean": float(block["case_mae"].mean()),
                "case_mae_std": float(block["case_mae"].std(ddof=0)),
                "case_mae_min": float(block["case_mae"].min()),
                "case_mae_median": float(block["case_mae"].median()),
                "case_mae_max": float(block["case_mae"].max()),
                "case_rmse_mean": float(block["case_rmse"].mean()),
                "case_rmse_std": float(block["case_rmse"].std(ddof=0)),
                "mean_signed_error_mean": float(block["mean_signed_error"].mean()),
                "mean_signed_error_std": float(block["mean_signed_error"].std(ddof=0)),
                "case_count": int(block["case_count"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae_mean", "case_mae_std", "variant_name"]).reset_index(drop=True)


def load_direct_2s_8s_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRY063_CASE_PATH.exists():
        raise FileNotFoundError(f"缺少 063 输出，无法复用 2s+8s 结果: {TRY063_CASE_PATH}")
    case_df = pd.read_csv(TRY063_CASE_PATH)
    rows: list[dict[str, object]] = []
    for domain_name in ("added", "added2", "all_external"):
        subset = case_df if domain_name == "all_external" else case_df.loc[case_df["domain"] == domain_name].copy()
        for model_name in ("2s", "8s", "2s_8s_fusion"):
            signed_error = subset[f"signed_error_{model_name}"].to_numpy(dtype=float, copy=False)
            abs_error = subset[f"abs_error_{model_name}"].to_numpy(dtype=float, copy=False)
            rows.append(
                {
                    "domain": domain_name,
                    "variant_name": f"direct_tinytcn_{model_name}_from_063",
                    "model_family": "tinytcn_direct",
                    "input_columns": "all_channels",
                    "strain_transform": "none",
                    "seed_count": 1,
                    "case_mae": float(abs_error.mean()),
                    "case_rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
                    "mean_signed_error": float(signed_error.mean()),
                    "case_count": int(len(subset)),
                }
            )
    added2_case_df = case_df.loc[case_df["domain"] == "added2"].copy()
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "variant_name"]).reset_index(drop=True), added2_case_df


def build_comparison_overview(stability_df: pd.DataFrame, direct_summary_df: pd.DataFrame) -> pd.DataFrame:
    midband_compare = stability_df.copy()
    midband_compare["source"] = "064_midband_multiseed"
    direct_compare = direct_summary_df.rename(
        columns={
            "case_mae": "case_mae_mean",
            "case_rmse": "case_rmse_mean",
            "mean_signed_error": "mean_signed_error_mean",
        }
    ).copy()
    direct_compare["source"] = "063_reused_direct"
    for column in ("fusion_weight_learned", "best_seed_count", "best_seed_rate", "case_mae_std", "case_rmse_std", "mean_signed_error_std"):
        if column not in direct_compare.columns:
            direct_compare[column] = np.nan
    common_columns = [
        "domain",
        "variant_name",
        "source",
        "model_family",
        "input_columns",
        "strain_transform",
        "fusion_weight_learned",
        "seed_count",
        "best_seed_count",
        "best_seed_rate",
        "case_mae_mean",
        "case_mae_std",
        "case_rmse_mean",
        "case_rmse_std",
        "mean_signed_error_mean",
        "mean_signed_error_std",
        "case_count",
    ]
    return pd.concat([midband_compare[common_columns], direct_compare[common_columns]], ignore_index=True).sort_values(
        ["domain", "case_mae_mean", "case_rmse_mean", "variant_name"]
    ).reset_index(drop=True)


def build_added2_case_comparison(seed_case_level_df: pd.DataFrame, direct_added2_case_df: pd.DataFrame) -> pd.DataFrame:
    added2_seed_df = seed_case_level_df.loc[seed_case_level_df["domain"] == "added2"].copy()
    rows: list[dict[str, object]] = []
    for (case_id, variant_name), block in added2_seed_df.groupby(["case_id", "variant_name"], sort=True):
        rows.append(
            {
                "case_id": int(case_id),
                "file_name": block["file_name"].iloc[0],
                "true_wind_speed": float(block["true_wind_speed"].iloc[0]),
                "rpm": float(block["rpm"].iloc[0]),
                "variant_name": variant_name,
                "pred_mean": float(block["pred_wind_speed"].mean()),
                "pred_std": float(block["pred_wind_speed"].std(ddof=0)),
                "signed_error_mean": float(block["signed_error"].mean()),
                "abs_error_mean": float(block["abs_error"].mean()),
                "seed_count": int(block["base_seed"].nunique()),
            }
        )
    for _, row in direct_added2_case_df.iterrows():
        for model_name in ("2s", "8s", "2s_8s_fusion"):
            pred = float(row[f"pred_{model_name}"])
            signed_error = float(pred - float(row["true_wind_speed"]))
            rows.append(
                {
                    "case_id": int(row["case_id"]),
                    "file_name": row["file_name"],
                    "true_wind_speed": float(row["true_wind_speed"]),
                    "rpm": float(row["rpm"]),
                    "variant_name": f"direct_tinytcn_{model_name}_from_063",
                    "pred_mean": pred,
                    "pred_std": np.nan,
                    "signed_error_mean": signed_error,
                    "abs_error_mean": abs(signed_error),
                    "seed_count": 1,
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "abs_error_mean", "variant_name"]).reset_index(drop=True)


def write_summary_markdown(
    output_path: Path,
    stability_df: pd.DataFrame,
    direct_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    added2_case_comparison_df: pd.DataFrame,
) -> None:
    lines = [
        "# added SOTA 回放 added2",
        "",
        "- 首次确认：`2026-04-08`",
        "- 最近复核：`2026-04-08`",
        "- 代码口径：`src/try/064_added_sota_added2_replay/`",
        "- `2s+8s` 口径：复用 `063` 的 full final deploy 输出，不在本脚本重训。",
        "- `midband` 口径：复用 `042` 的 `rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz)` 和 10 个 seed。",
        "- 信号列口径：为同时评估 `added2`，取 `final / added / added2` 的共同列。",
        "",
        "## added2 汇总",
        "",
    ]
    added2_compare = comparison_df.loc[comparison_df["domain"] == "added2"].copy()
    for _, row in added2_compare.iterrows():
        std_text = "" if pd.isna(row["case_mae_std"]) else f", case_mae_std=`{row['case_mae_std']:.4f}`"
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae_mean']:.4f}`{std_text}, "
            f"mean_signed_error=`{row['mean_signed_error_mean']:+.4f}`, source=`{row['source']}`"
        )
    lines.extend(["", "## added 复核", ""])
    added_compare = comparison_df.loc[comparison_df["domain"] == "added"].copy()
    for _, row in added_compare.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae_mean']:.4f}`, "
            f"mean_signed_error=`{row['mean_signed_error_mean']:+.4f}`, source=`{row['source']}`"
        )
    lines.extend(["", "## added2 每工况重点", ""])
    focus_variants = {
        "rpm_knn4",
        "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3",
        "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5",
        "direct_tinytcn_2s_8s_fusion_from_063",
    }
    for case_id, block in added2_case_comparison_df.groupby("case_id", sort=True):
        lines.append(f"### 工况{int(case_id)}")
        lines.append("")
        for _, row in block.loc[block["variant_name"].isin(focus_variants)].sort_values(["abs_error_mean", "variant_name"]).iterrows():
            lines.append(
                f"- `{row['variant_name']}`: true=`{row['true_wind_speed']:.4f}`, "
                f"pred=`{row['pred_mean']:.4f}`, abs_error=`{row['abs_error_mean']:.4f}`"
            )
        lines.append("")
    best_row = added2_compare.iloc[0]
    midband_w03 = added2_compare.loc[added2_compare["variant_name"] == "fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3"].iloc[0]
    direct_fusion = added2_compare.loc[added2_compare["variant_name"] == "direct_tinytcn_2s_8s_fusion_from_063"].iloc[0]
    lines.extend(
        [
            "## 当前判断",
            "",
            f"- `2026-04-08` 按 added 上界 SOTA 口径回放到 added2 后，最优 added2 候选是 `{best_row['variant_name']}`，case_mae=`{best_row['case_mae_mean']:.4f}`。",
            f"- `rpm_knn4 + midband @ w=0.3` 在 added2 的 case_mae=`{midband_w03['case_mae_mean']:.4f}`，不如直接 `rpm_knn4` 稳。",
            f"- 复用 `063` 的 direct `2s+8s` 在 added2 的 case_mae=`{direct_fusion['case_mae_mean']:.4f}`，仍明显弱于 rpm-first 路线。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
