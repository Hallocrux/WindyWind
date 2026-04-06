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
VENDOR_ROOT = TRY_ROOT / "vendor_samformer" / "samformer_pytorch"
for candidate in (TRY015_ROOT, VENDOR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from samformer.samformer import SAMFormerArchitecture

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

TRY_NAME = "017_samformer_loco"
MODEL_NAME = "SAMformer"
OFFICIAL_REPO_URL = "https://github.com/romilbert/samformer"
OFFICIAL_COMMIT = "71f10eaa696f2a098798779ee14b6ecd6b69bcd9"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME


class SAMFormerWindRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        *,
        hid_dim: int,
        use_revin: bool,
    ) -> None:
        super().__init__()
        self.backbone = SAMFormerArchitecture(
            num_channels=in_channels,
            seq_len=seq_len,
            hid_dim=hid_dim,
            pred_horizon=1,
            use_revin=use_revin,
        )
        self.readout = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x, flatten_output=False).squeeze(-1)
        return self.readout(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAMformer 官方骨干在 windywind 上的 LOCO 实验。")
    add_common_args(parser, default_output_dir=OUTPUT_ROOT)
    parser.add_argument("--hid-dim", type=int, default=16)
    parser.add_argument(
        "--disable-revin",
        action="store_true",
        help="关闭官方 SAMformer 中的 RevIN。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(args)
    dataset = build_dataset_from_records(records)
    train_config = build_train_config(args)

    def build_model(in_channels: int, seq_len: int, _device: torch.device) -> nn.Module:
        return SAMFormerWindRegressor(
            in_channels=in_channels,
            seq_len=seq_len,
            hid_dim=args.hid_dim,
            use_revin=not args.disable_revin,
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
