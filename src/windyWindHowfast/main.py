from __future__ import annotations

from .app import build_arg_parser, run_analysis
from .analysis_core import resolve_video_path


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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


__all__ = ["build_arg_parser", "main", "run_analysis"]
