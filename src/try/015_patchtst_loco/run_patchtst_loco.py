from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

TRY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRY_ROOT.parents[2]
VENDOR_ROOT = TRY_ROOT / "vendor_PatchTST" / "PatchTST_supervised"
if str(TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY_ROOT))
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from models.PatchTST import Model as PatchTSTForecastModel

from windywind_official_common import (
    build_dataset_from_records,
    build_train_config,
    add_common_args,
    evaluate_scalar_model_loco,
    predict_scalar_model_unlabeled,
    select_records,
    write_run_metadata,
    write_summary_markdown,
)
from phase3_cnn_tcn_lib import summarize_predictions

TRY_NAME = "015_patchtst_loco"
MODEL_NAME = "PatchTST"
OFFICIAL_REPO_URL = "https://github.com/yuqinie98/PatchTST"
OFFICIAL_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME


class PatchTSTWindRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        *,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        patch_len: int,
        stride: int,
        dropout: float,
        fc_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        configs = SimpleNamespace(
            enc_in=in_channels,
            seq_len=seq_len,
            pred_len=1,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            individual=False,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            revin=True,
            affine=True,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.backbone = PatchTSTForecastModel(configs)
        self.readout = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast = self.backbone(x.transpose(1, 2))
        return self.readout(forecast[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 PatchTST 官方骨干在 windywind 上的 LOCO 实验。")
    add_common_args(parser, default_output_dir=OUTPUT_ROOT)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--patch-len", type=int, default=25)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(args)
    dataset = build_dataset_from_records(records)
    train_config = build_train_config(args)

    def build_model(in_channels: int, seq_len: int, _device: torch.device) -> nn.Module:
        return PatchTSTWindRegressor(
            in_channels=in_channels,
            seq_len=seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            patch_len=args.patch_len,
            stride=args.stride,
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,
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
