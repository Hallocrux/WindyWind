from __future__ import annotations

import argparse
from pathlib import Path

from .animation import save_mode_shape_animations
from .models import ModalConfig
from .pipeline import DEFAULT_OUTPUT_DIR, run_modal_identification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行模态参数识别模块。")
    parser.add_argument("--case-ids", nargs="+", type=int, default=None, help="只运行指定工况编号。")
    parser.add_argument("--freq-min", type=float, default=0.5, help="预分析频率下限。")
    parser.add_argument("--freq-max", type=float, default=6.0, help="预分析频率上限。")
    parser.add_argument("--window-seconds", type=float, default=20.0, help="SSI 窗口长度（秒）。")
    parser.add_argument("--step-seconds", type=float, default=10.0, help="SSI 步长（秒）。")
    parser.add_argument("--rpm-source", choices=("manifest", "sync_csv"), default="manifest", help="rpm 入口类型。")
    parser.add_argument("--rpm-series-path", type=Path, default=None, help="同步 rpm 时程 CSV 路径。")
    parser.add_argument("--harmonic-orders", nargs="+", type=int, default=[1, 2, 3, 4], help="需要标记和屏蔽的谐波阶次。")
    parser.add_argument("--harmonic-half-width", type=float, default=0.2, help="每个谐波屏蔽半宽（Hz）。")
    parser.add_argument("--sensor-basis", choices=("strain", "acc_y", "both"), default="both", help="默认输出的模态测点基底。")
    parser.add_argument("--fe-reference-path", type=Path, default=None, help="外部 FE/梁模型参考 CSV。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--save-mode-shape-animation", action="store_true", help="显式导出振型动画。")
    parser.add_argument("--animation-format", choices=("auto", "mp4", "gif"), default="auto", help="动画写出格式。")
    parser.add_argument("--animation-fps", type=int, default=15, help="动画帧率。")
    parser.add_argument("--animation-cycles", type=int, default=2, help="动画播放周期数。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModalConfig(
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        harmonic_orders=tuple(args.harmonic_orders),
        harmonic_half_width=args.harmonic_half_width,
    )
    outputs = run_modal_identification(
        case_ids=args.case_ids,
        output_dir=args.output_dir,
        sensor_basis=args.sensor_basis,
        rpm_source=args.rpm_source,
        rpm_series_path=args.rpm_series_path,
        fe_reference_path=args.fe_reference_path,
        config=config,
    )
    summary = outputs["case_modal_summary"]
    print(f"已输出目录: {args.output_dir}")
    if not summary.empty:
        print(
            summary[
                [
                    "case_id",
                    "strain_first_frequency_hz",
                    "strain_damping_ratio",
                    "accy_first_frequency_hz",
                    "accy_damping_ratio",
                ]
            ].to_string(index=False)
        )
    if not args.save_mode_shape_animation:
        return

    animation_rows = save_mode_shape_animations(
        output_dir=args.output_dir,
        shape_tables={
            "strain": outputs["strain_mode_shapes"],
            "accy": outputs["accy_mode_shapes"],
        },
        animation_format=args.animation_format,
        fps=args.animation_fps,
        cycles=args.animation_cycles,
    )
    for row in animation_rows:
        case_text = "N/A" if row["case_id"] is None else f"{int(row['case_id']):02d}"
        print(
            f"animation case={case_text} basis={row['basis']} status={row['status']} "
            f"message={row['message']} path={row['path'] or '-'}"
        )


if __name__ == "__main__":
    main()
