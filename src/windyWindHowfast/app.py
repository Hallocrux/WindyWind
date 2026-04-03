from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .analysis_core import (
    analyze_spatiotemporal_frequency,
    build_angle_profile,
    render_summary_figure,
    resolve_output_dir,
    resolve_video_path,
)
from .constants import (
    DEFAULT_ANGULAR_RESOLUTION,
    DEFAULT_ROI_FRAME_STRATEGY,
    DEFAULT_ROI_REFERENCE_MAX_FRAMES,
    DEFAULT_ROI_SCORE_THRESHOLD,
)
from .models import AnalysisResult
from .outputs import save_analysis_outputs, save_roi_debug_outputs
from .roi_detection import resolve_roi_candidate, save_roi_config


def run_analysis(
    video_path: Path,
    angular_res: int = DEFAULT_ANGULAR_RESOLUTION,
    min_temporal_hz: float = 0.2,
    max_temporal_hz: float = 20.0,
    max_spatial_mode: int = 24,
    inner_radius_ratio: float = 0.25,
    center_x: int | None = None,
    center_y: int | None = None,
    radius: int | None = None,
    roi_path: str | None = None,
    save_roi_path: str | None = None,
    output_dir: str | None = None,
    run_name: str = "analysis",
    show_plot: bool = True,
    start_frame: int = 0,
    max_frames: int | None = None,
    auto_roi: bool = True,
    interactive: bool = True,
    roi_frame_strategy: str = DEFAULT_ROI_FRAME_STRATEGY,
    roi_reference_max_frames: int = DEFAULT_ROI_REFERENCE_MAX_FRAMES,
    roi_score_threshold: float = DEFAULT_ROI_SCORE_THRESHOLD,
    roi_debug: bool = True,
) -> AnalysisResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("无法读取视频首帧。")

    resolved_output_dir = resolve_output_dir(output_dir, video_path)

    print(f"视频路径: {video_path}")
    print(f"FPS: {fps:.2f}, 总帧数: {total_frames}")

    roi_config, selection_result = resolve_roi_candidate(
        cap=cap,
        first_frame=first_frame,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        roi_path=roi_path,
        auto_roi=auto_roi,
        interactive=interactive,
        roi_frame_strategy=roi_frame_strategy,
        roi_reference_max_frames=roi_reference_max_frames,
        roi_score_threshold=roi_score_threshold,
    )
    save_roi_debug_outputs(
        output_dir=resolved_output_dir,
        run_name=run_name,
        first_frame=first_frame,
        selection_result=selection_result,
        roi_debug=roi_debug,
    )

    if roi_config is None:
        cap.release()
        raise RuntimeError(f"ROI 获取失败: {selection_result.failure_reason or 'unknown_failure'}")

    if roi_config.radius < 8:
        cap.release()
        raise ValueError("选择的分析半径过小。")

    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"start-frame 超出范围: {start_frame}")

    if save_roi_path:
        save_roi_config(roi_config, Path(save_roi_path))
        print(f"已保存 ROI 参数到: {save_roi_path}")

    selected_score = selection_result.selected_score.total_score if selection_result.selected_score else None
    print(
        "ROI 结果："
        f"source={selection_result.source}, "
        f"status={selection_result.status}, "
        f"score={selected_score if selected_score is not None else 'n/a'}"
    )
    print(
        "开始处理视频，"
        f"中心=({roi_config.center_x}, {roi_config.center_y})，"
        f"半径={roi_config.radius}px，角度分辨率={angular_res}，"
        f"起始帧={start_frame}"
    )

    frames_to_process = total_frames - start_frame
    if max_frames is not None:
        frames_to_process = min(frames_to_process, max_frames)
    if frames_to_process < 4:
        cap.release()
        raise ValueError("分析帧数不足，至少需要 4 帧。")

    time_angle_map = np.zeros((frames_to_process, angular_res), dtype=np.float32)
    debug_edges = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 0
    while cap.isOpened() and frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        angle_profile, edges = build_angle_profile(
            gray,
            (roi_config.center_x, roi_config.center_y),
            roi_config.radius,
            angular_res,
            inner_radius_ratio=inner_radius_ratio,
        )
        time_angle_map[frame_idx] = angle_profile
        if debug_edges is None:
            debug_edges = edges

        if frame_idx % 50 == 0:
            print(f"已处理帧: {frame_idx}/{frames_to_process}")

        frame_idx += 1

    cap.release()
    time_angle_map = time_angle_map[:frame_idx]

    (
        rotor_freq_hz,
        peak_temporal_hz,
        peak_spatial_mode,
        peak_magnitude,
        processed_map,
        spectrum_mag,
        temporal_freqs,
        spatial_modes,
    ) = analyze_spatiotemporal_frequency(
        time_angle_map=time_angle_map,
        fps=fps,
        min_temporal_hz=min_temporal_hz,
        max_temporal_hz=max_temporal_hz,
        max_spatial_mode=max_spatial_mode,
    )

    rpm = rotor_freq_hz * 60.0
    print()
    print("=" * 20)
    print(f"主导时间频率: {peak_temporal_hz:.2f} Hz")
    print(f"主导角向模态: k = {peak_spatial_mode}")
    print(f"谱峰幅值: {peak_magnitude:.2f}")
    print(f"估计转子频率: {rotor_freq_hz:.2f} Hz")
    print(f"估计转速: {rpm:.2f} RPM")
    print("=" * 20)

    figure = render_summary_figure(
        time_angle_map=time_angle_map,
        debug_edges=debug_edges,
        processed_map=processed_map,
        spectrum_mag=spectrum_mag,
        temporal_freqs=temporal_freqs,
        spatial_modes=spatial_modes,
        frame_idx=frame_idx,
        fps=fps,
        peak_temporal_hz=peak_temporal_hz,
        peak_spatial_mode=peak_spatial_mode,
    )

    result = AnalysisResult(
        video_path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        start_frame=start_frame,
        analyzed_frames=frame_idx,
        analyzed_duration_sec=frame_idx / fps,
        angular_res=angular_res,
        inner_radius_ratio=inner_radius_ratio,
        min_temporal_hz=min_temporal_hz,
        max_temporal_hz=max_temporal_hz,
        max_spatial_mode=max_spatial_mode,
        roi=asdict(roi_config),
        roi_source=selection_result.source,
        roi_detection_status=selection_result.status,
        roi_score=selected_score,
        roi_reference_frame_strategy=selection_result.reference_frame_strategy,
        roi_reference_frame_indices=selection_result.reference_frame_indices,
        peak_temporal_hz=peak_temporal_hz,
        peak_spatial_mode=peak_spatial_mode,
        peak_magnitude=peak_magnitude,
        rotor_freq_hz=rotor_freq_hz,
        rpm=rpm,
    )

    save_analysis_outputs(
        output_dir=resolved_output_dir,
        run_name=run_name,
        first_frame=first_frame,
        roi_config=roi_config,
        figure=figure,
        result=result,
    )
    print(f"结果已保存到: {resolved_output_dir}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于极坐标展开与二维时空频谱的风机视频转速分析。"
    )
    parser.add_argument("--video", type=str, default=None, help="视频路径。未指定时默认读取 data/video 下的第一个 mp4。")
    parser.add_argument("--angular-res", type=int, default=DEFAULT_ANGULAR_RESOLUTION, help="角度离散分辨率。")
    parser.add_argument("--min-temporal-hz", type=float, default=0.2, help="时间频率搜索下界。")
    parser.add_argument("--max-temporal-hz", type=float, default=20.0, help="时间频率搜索上界。")
    parser.add_argument("--max-spatial-mode", type=int, default=24, help="空间角向模态搜索上界。")
    parser.add_argument("--inner-radius-ratio", type=float, default=0.25, help="极坐标图中忽略中心附近半径的比例，用于减少轮毂/支撑杆干扰。")
    parser.add_argument("--center-x", type=int, default=None, help="ROI 中心点 x 坐标。")
    parser.add_argument("--center-y", type=int, default=None, help="ROI 中心点 y 坐标。")
    parser.add_argument("--radius", type=int, default=None, help="ROI 半径。")
    parser.add_argument("--roi-json", type=str, default=None, help="从 JSON 文件加载 ROI 参数，字段为 center_x / center_y / radius。")
    parser.add_argument("--save-roi-json", type=str, default=None, help="把本次使用的 ROI 参数写到 JSON 文件，便于后续复用。")
    parser.add_argument("--auto-roi", action=argparse.BooleanOptionalAction, default=True, help="是否启用自动 ROI 候选框架。")
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True, help="自动 ROI 失败时是否允许回退到手动点选。")
    parser.add_argument("--roi-frame-strategy", choices=["early_stable", "uniform_prefix", "low_motion_subset", "manual_index_list"], default=DEFAULT_ROI_FRAME_STRATEGY, help="ROI 参考帧采样策略。")
    parser.add_argument("--roi-reference-max-frames", type=int, default=DEFAULT_ROI_REFERENCE_MAX_FRAMES, help="构建 ROI 候选时最多采样多少参考帧。")
    parser.add_argument("--roi-score-threshold", type=float, default=DEFAULT_ROI_SCORE_THRESHOLD, help="自动 ROI 候选最低通过分数。")
    parser.add_argument("--roi-debug", action=argparse.BooleanOptionalAction, default=True, help="是否输出候选级 ROI debug 工件。")
    parser.add_argument("--output-dir", type=str, default=None, help="结果输出目录。默认写到 outputs/windyWindHowfast/<视频名>/。")
    parser.add_argument("--run-name", type=str, default="analysis", help="本次运行名称，用于区分多次测试输出文件。")
    parser.add_argument("--no-show", action="store_true", help="只落盘结果，不弹出 matplotlib 窗口。")
    parser.add_argument("--start-frame", type=int, default=0, help="从指定帧开始分析，便于跳过前段扰动。")
    parser.add_argument("--max-frames", type=int, default=None, help="仅处理前 N 帧，便于快速测试参数与流程。")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    video_path = resolve_video_path(args.video)
    run_analysis(
        video_path=video_path,
        angular_res=args.angular_res,
        min_temporal_hz=args.min_temporal_hz,
        max_temporal_hz=args.max_temporal_hz,
        max_spatial_mode=args.max_spatial_mode,
        inner_radius_ratio=args.inner_radius_ratio,
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        roi_path=args.roi_json,
        save_roi_path=args.save_roi_json,
        output_dir=args.output_dir,
        run_name=args.run_name,
        show_plot=not args.no_show,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        auto_roi=args.auto_roi,
        interactive=args.interactive,
        roi_frame_strategy=args.roi_frame_strategy,
        roi_reference_max_frames=args.roi_reference_max_frames,
        roi_score_threshold=args.roi_score_threshold,
        roi_debug=args.roi_debug,
    )
