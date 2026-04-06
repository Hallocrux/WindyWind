from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

TRY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRY_ROOT.parents[2]
TRY015_ROOT = REPO_ROOT / "src" / "try" / "015_patchtst_loco"
VENDOR_ROOT = TRY_ROOT / "vendor_MICN"
for candidate in (TRY015_ROOT, VENDOR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from models.model import MICN

from windywind_official_common import (
    add_common_args,
    build_dataset_from_records,
    build_train_config,
    evaluate_scalar_model_loco,
    predict_scalar_model_unlabeled,
    select_records,
    write_run_metadata,
    write_summary_markdown,
)
from phase3_cnn_tcn_lib import summarize_predictions

TRY_NAME = "016_micn_loco"
MODEL_NAME = "MICN"
OFFICIAL_REPO_URL = "https://github.com/wanghq21/MICN"
OFFICIAL_COMMIT = "370c69b841d72246556ca05dd23163c560c22b5a"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME


class MICNWindRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        *,
        d_model: int,
        n_heads: int,
        d_layers: int,
        dropout: float,
        device: torch.device,
        conv_kernel_small: int,
        conv_kernel_large: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.time_feature_dim = 4
        pred_len = 1
        conv_kernel = [conv_kernel_small, conv_kernel_large]
        decomp_kernel: list[int] = []
        isometric_kernel: list[int] = []
        for kernel_size in conv_kernel:
            if kernel_size % 2 == 0:
                decomp_kernel.append(kernel_size + 1)
                isometric_kernel.append((seq_len + pred_len + kernel_size) // kernel_size)
            else:
                decomp_kernel.append(kernel_size)
                isometric_kernel.append((seq_len + pred_len + kernel_size - 1) // kernel_size)
        self.model = MICN(
            dec_in=in_channels,
            c_out=1,
            seq_len=seq_len,
            label_len=seq_len,
            out_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            d_layers=d_layers,
            dropout=dropout,
            embed="fixed",
            freq="h",
            device=device,
            mode="regre",
            decomp_kernel=decomp_kernel,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_enc = x.transpose(1, 2)
        x_mark_enc = torch.zeros(
            batch_size,
            self.seq_len,
            self.time_feature_dim,
            device=x.device,
        )
        x_dec = torch.zeros(
            batch_size,
            self.seq_len + 1,
            self.in_channels,
            device=x.device,
        )
        x_mark_dec = torch.zeros(
            batch_size,
            self.seq_len + 1,
            self.time_feature_dim,
            device=x.device,
        )
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -1, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MICN 官方骨干在 windywind 上的 LOCO 实验。")
    add_common_args(parser, default_output_dir=OUTPUT_ROOT)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv-kernel-small", type=int, default=12)
    parser.add_argument("--conv-kernel-large", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(args)
    dataset = build_dataset_from_records(records)
    train_config = build_train_config(args)

    def build_model(in_channels: int, seq_len: int, device: torch.device) -> nn.Module:
        return MICNWindRegressor(
            in_channels=in_channels,
            seq_len=seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_layers=args.d_layers,
            dropout=args.dropout,
            device=device,
            conv_kernel_small=args.conv_kernel_small,
            conv_kernel_large=args.conv_kernel_large,
        )

    prediction_frame = evaluate_scalar_model_loco(
        dataset,
        build_model,
        train_config,
        random_seed=args.random_seed,
    )
    summary_row, case_df = summarize_predictions(prediction_frame, MODEL_NAME)
    summary_df = pd.DataFrame([summary_row])
    unlabeled_df = predict_scalar_model_unlabeled(
        dataset,
        build_model,
        train_config,
        random_seed=args.random_seed,
    )

    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    unlabeled_df.to_csv(output_dir / "unlabeled_predictions.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_dir,
        model_name=MODEL_NAME,
        summary_row=summary_row,
        records=records,
        official_repo_url=OFFICIAL_REPO_URL,
        official_commit=OFFICIAL_COMMIT,
    )
    write_run_metadata(
        output_dir,
        model_name=MODEL_NAME,
        args=args,
        records=records,
        official_repo_url=OFFICIAL_REPO_URL,
        official_commit=OFFICIAL_COMMIT,
    )

    print(f"{MODEL_NAME} 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"case_mae: {summary_row['case_mae']:.4f}")


if __name__ == "__main__":
    main()
