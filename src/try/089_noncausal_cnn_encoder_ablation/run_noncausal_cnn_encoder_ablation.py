from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "089_noncausal_cnn_encoder_ablation"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
COMMON_ROOT = REPO_ROOT / "src" / "try" / "066_reuse_embedding_domain_split"
TRY071_SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "try"
    / "071_external_embedding_regression_quickcheck"
    / "run_external_embedding_regression_quickcheck.py"
)
WINDOW_LABELS = ("2s", "8s")
EMBEDDING_DIM_PER_WINDOW = 32

if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reuse_embedding_domain_common import (  # noqa: E402
    build_cleaned_signal_frames,
    build_embedding_case_table,
    build_record_table,
    load_source_catalog,
    load_try053_module,
)


@dataclass(frozen=True)
class EncoderVariant:
    name: str
    with_dilation: bool

    @property
    def dilations(self) -> tuple[int, int, int]:
        return (1, 2, 4) if self.with_dilation else (1, 1, 1)


class NonCausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int) -> None:
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
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out + self.downsample(x)


class NonCausalCNNEncoderRegressor(nn.Module):
    def __init__(self, in_channels: int, *, with_dilation: bool) -> None:
        super().__init__()
        dilations = (1, 2, 4) if with_dilation else (1, 1, 1)
        self.blocks = nn.Sequential(
            NonCausalResidualBlock(in_channels, 16, dilation=dilations[0]),
            NonCausalResidualBlock(16, EMBEDDING_DIM_PER_WINDOW, dilation=dilations[1]),
            NonCausalResidualBlock(EMBEDDING_DIM_PER_WINDOW, EMBEDDING_DIM_PER_WINDOW, dilation=dilations[2]),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(EMBEDDING_DIM_PER_WINDOW, 16),
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
    parser = argparse.ArgumentParser(description="Non-causal 1D CNN encoder ablation for 071 downstream ridge.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try053 = load_try053_module()
    try071 = load_try071_module()
    train_config = try053.TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    catalog = load_source_catalog()
    train_records = sorted(
        [
            *[record for record in catalog.final_records if record.is_labeled],
            *[record for record in catalog.added_records if record.is_labeled],
        ],
        key=lambda record: record.case_id,
    )
    export_records = catalog.all_records
    record_df = build_record_table(catalog)
    _, cleaned_signal_frames = build_cleaned_signal_frames(export_records)

    variants = (
        EncoderVariant("noncausal_dilated", with_dilation=True),
        EncoderVariant("noncausal_nodilation", with_dilation=False),
    )
    summary_frames: list[pd.DataFrame] = []
    summary_domain_frames: list[pd.DataFrame] = []
    for variant_idx, variant in enumerate(variants, start=1):
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        per_window = {}
        for window_idx, window_label in enumerate(WINDOW_LABELS, start=1):
            per_window[window_label] = load_or_train_window_embeddings(
                try053=try053,
                variant=variant,
                train_records=train_records,
                export_records=export_records,
                cleaned_signal_frames=cleaned_signal_frames,
                window_config=try053.WINDOW_CONFIGS[window_label],
                window_label=window_label,
                seed=args.random_seed + variant_idx * 10000 + window_idx * 1000,
                train_config=train_config,
                checkpoint_dir=variant_dir / "models" / "checkpoints",
                force_retrain=args.force_retrain,
            )

        embedding_case_df = build_embedding_case_table(record_df, per_window)
        embedding_columns = [column for column in embedding_case_df.columns if column.startswith("embedding_")]
        if len(embedding_columns) != EMBEDDING_DIM_PER_WINDOW * len(WINDOW_LABELS):
            raise RuntimeError(f"{variant.name} embedding 维度异常: {len(embedding_columns)}")
        embedding_case_df.to_csv(variant_dir / "embedding_case_table.csv", index=False, encoding="utf-8-sig")

        summary_df, summary_domain_df = run_071_downstream_eval(
            try071=try071,
            embedding_case_df=embedding_case_df,
            output_dir=variant_dir,
        )
        summary_df.insert(0, "encoder_variant", variant.name)
        summary_domain_df.insert(0, "encoder_variant", variant.name)
        summary_frames.append(summary_df)
        summary_domain_frames.append(summary_domain_df)

    combined_summary_df = pd.concat(summary_frames, ignore_index=True)
    combined_domain_df = pd.concat(summary_domain_frames, ignore_index=True)
    combined_summary_df.to_csv(output_dir / "combined_summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    combined_domain_df.to_csv(output_dir / "combined_summary_by_protocol_and_domain.csv", index=False, encoding="utf-8-sig")
    write_top_summary(output_dir / "summary.md", combined_summary_df)

    main_block = combined_summary_df.loc[combined_summary_df["protocol"] == "added_to_added2"].copy()
    best = main_block.sort_values(["case_mae", "case_rmse", "encoder_variant", "variant_name"]).iloc[0]
    print("089 non-causal CNN encoder ablation 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best added_to_added2: encoder={best['encoder_variant']} | "
        f"variant={best['variant_name']} | case_mae={best['case_mae']:.4f}"
    )


def load_try071_module():
    spec = importlib.util.spec_from_file_location("try071_external_embedding_regression", TRY071_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 071 脚本: {TRY071_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_or_train_window_embeddings(
    *,
    try053,
    variant: EncoderVariant,
    train_records,
    export_records,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config,
    window_label: str,
    seed: int,
    train_config,
    checkpoint_dir: Path,
    force_retrain: bool,
) -> dict[str, object]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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

    x_train = train_dataset.windows.astype(np.float32)
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    x_export = export_dataset.windows.astype(np.float32)

    ckpt_base = checkpoint_dir / f"{variant.name}_{window_label}"
    ckpt_path = ckpt_base.with_suffix(".pt")
    norm_path = checkpoint_dir / f"{variant.name}_{window_label}_norm.npz"
    meta_path = checkpoint_dir / f"{variant.name}_{window_label}.json"

    model = NonCausalCNNEncoderRegressor(
        in_channels=x_train.shape[1],
        with_dilation=variant.with_dilation,
    )
    if ckpt_path.exists() and norm_path.exists() and meta_path.exists() and not force_retrain:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        norm = np.load(norm_path)
        mean = norm["mean"]
        std = norm["std"]
    else:
        x_train_norm, _, mean, std = try053.normalize_windows_by_channel(x_train, x_train)
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu")
        model = model.to(device)
        try053.train_model(model, x_train_norm, y_train, train_config, device)
        torch.save(model.cpu().state_dict(), ckpt_path)
        np.savez(norm_path, mean=mean, std=std)
        meta_path.write_text(
            json.dumps(
                {
                    "encoder_variant": variant.name,
                    "with_dilation": variant.with_dilation,
                    "dilations": list(variant.dilations),
                    "window_label": window_label,
                    "embedding_dim_per_window": EMBEDDING_DIM_PER_WINDOW,
                    "seed": seed,
                    "train_case_ids": [int(record.case_id) for record in train_records],
                    "export_case_ids": [int(record.case_id) for record in export_records],
                    "architecture_note": "non-causal symmetric padding; residual block; two Conv1d layers per block; global average pooling",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    model.eval()
    x_export_norm = ((x_export - mean) / std).astype(np.float32)
    with torch.no_grad():
        export_tensor = torch.from_numpy(x_export_norm).float()
        export_embedding = model.encode(export_tensor).cpu().numpy()
    if export_embedding.shape[1] != EMBEDDING_DIM_PER_WINDOW:
        raise RuntimeError(f"{variant.name} {window_label} embedding 维度异常: {export_embedding.shape}")
    return {
        "export_meta_df": export_dataset.meta_df.copy().reset_index(drop=True),
        "window_embedding": export_embedding,
        "mean": mean,
        "std": std,
    }


def run_071_downstream_eval(
    *,
    try071,
    embedding_case_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    external_df = (
        embedding_case_df.loc[
            embedding_case_df["raw_source_domain"].isin(["added", "added2"])
            & embedding_case_df["is_labeled"]
        ]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    embedding_columns = [column for column in external_df.columns if column.startswith("embedding_")]
    loocv_pred_df = try071.run_external_loocv(external_df, embedding_columns)
    transfer_pred_df = try071.run_added_to_added2_transfer(external_df, embedding_columns)
    all_pred_df = pd.concat([loocv_pred_df, transfer_pred_df], ignore_index=True)

    summary_df = try071.build_summary_by_protocol(all_pred_df)
    summary_domain_df = try071.build_summary_by_protocol_and_domain(all_pred_df)
    all_pred_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_domain_df.to_csv(output_dir / "summary_by_protocol_and_domain.csv", index=False, encoding="utf-8-sig")
    try071.create_pred_vs_true_plot(
        all_pred_df.loc[all_pred_df["protocol"] == "external_loocv"].copy(),
        plot_dir / "pred_vs_true_external_loocv.png",
        "External LOOCV",
    )
    try071.create_pred_vs_true_plot(
        all_pred_df.loc[all_pred_df["protocol"] == "added_to_added2"].copy(),
        plot_dir / "pred_vs_true_added_to_added2.png",
        "Added to Added2",
    )
    try071.write_summary_markdown(output_dir / "summary.md", summary_df, summary_domain_df)
    return summary_df, summary_domain_df


def write_top_summary(output_path: Path, combined_summary_df: pd.DataFrame) -> None:
    lines = [
        "# non-causal CNN encoder ablation",
        "",
        "- 状态：`current`",
        "- 首次确认：`2026-04-13`",
        "- 最近复核：`2026-04-13`",
        "- 代码口径：`src/try/089_noncausal_cnn_encoder_ablation/`",
        "",
        "## Summary By Protocol",
        "",
    ]
    for protocol, protocol_df in combined_summary_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        ordered = protocol_df.sort_values(["case_mae", "case_rmse", "encoder_variant", "variant_name"])
        for _, row in ordered.iterrows():
            lines.append(
                f"- `{row['encoder_variant']} | {row['variant_name']}`: "
                f"case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, "
                f"mean_signed_error=`{row['mean_signed_error']:.4f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
