from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from task_io import load_task
    from video_rpm_eval import evaluate_task_video_rpm
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.windNotFound.task_io import load_task
    from src.windNotFound.video_rpm_eval import evaluate_task_video_rpm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="windNotFound 逐帧 ROI 视频 RPM 验证入口。")
    parser.add_argument("--task", required=True, help="标注任务 YAML 路径。")
    parser.add_argument("--selector-index", type=int, required=True, help="要验证的 selector 序号。")
    parser.add_argument("--angular-res", type=int, default=720, help="角度离散分辨率。")
    parser.add_argument("--inner-radius-ratio", type=float, default=0.25, help="忽略中心附近半径比例。")
    parser.add_argument("--min-temporal-hz", type=float, default=0.2, help="时间频率搜索下界。")
    parser.add_argument("--max-temporal-hz", type=float, default=20.0, help="时间频率搜索上界。")
    parser.add_argument("--max-spatial-mode", type=int, default=24, help="空间角向模态搜索上界。")
    args = parser.parse_args(argv)

    result = evaluate_task_video_rpm(
        load_task(args.task),
        selector_index=args.selector_index,
        angular_res=args.angular_res,
        inner_radius_ratio=args.inner_radius_ratio,
        min_temporal_hz=args.min_temporal_hz,
        max_temporal_hz=args.max_temporal_hz,
        max_spatial_mode=args.max_spatial_mode,
    )
    payload = result["payload"]["result"]
    print(f"评估结果: {result['output_path']}")
    print(f"selector={payload['selector_index']} status={payload['status']}")
    if payload["status"] == "ok":
        print(f"manual_rpm={payload['manual_fit']['rpm']}")
        print(f"video_rpm={payload['video_fft']['rpm']}")
        print(f"raw_max_peak_rpm={payload['video_fft']['raw_max_peak_rpm']}")
        print(f"selected_k={payload['video_fft']['peak_spatial_mode']}")
        print(f"abs_rpm_gap={payload['abs_rpm_gap']}")
        print("top_peaks:")
        for peak in payload["video_fft"].get("top_peaks", []):
            print(
                f"  rank={peak['rank']} "
                f"f={peak['temporal_hz']:.6f}Hz "
                f"k={peak['spatial_mode_k']} "
                f"mag={peak['magnitude']:.3f} "
                f"rpm={peak['rpm']:.3f}"
            )


if __name__ == "__main__":
    main()
