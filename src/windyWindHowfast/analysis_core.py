from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .constants import DEFAULT_ANGULAR_RESOLUTION, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEO_DIR


def build_angle_profile(
    gray: np.ndarray,
    center: tuple[int, int],
    radius: int,
    angular_res: int,
    inner_radius_ratio: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """把图像展开到极坐标后，仅压缩半径维，保留角度维结构。"""
    polar_img = cv2.warpPolar(
        gray,
        (radius, angular_res),
        center,
        radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    polar_blur = cv2.GaussianBlur(polar_img, (5, 5), 0)
    edges = cv2.Canny(polar_blur, 30, 100)

    inner_radius = min(max(int(radius * inner_radius_ratio), 0), radius - 1)
    radial_band = edges[:, inner_radius:]
    angle_profile = np.sum(radial_band, axis=1).astype(np.float32)
    return angle_profile, edges


def analyze_spatiotemporal_frequency(
    time_angle_map: np.ndarray,
    fps: float,
    min_temporal_hz: float = 0.2,
    max_temporal_hz: float = 20.0,
    max_spatial_mode: int = 24,
) -> tuple[float, float, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从完整的 time-angle map 中估计转子频率。"""
    if time_angle_map.shape[0] < 4:
        raise ValueError("至少需要 4 帧才能做时空频谱分析。")

    data = time_angle_map.astype(np.float32).copy()
    data -= np.mean(data, axis=1, keepdims=True)
    data -= np.mean(data, axis=0, keepdims=True)

    frame_std = np.std(data, axis=1, keepdims=True)
    data /= np.maximum(frame_std, 1e-6)

    time_window = np.hanning(data.shape[0]).astype(np.float32)[:, None]
    angle_window = np.hanning(data.shape[1]).astype(np.float32)[None, :]
    windowed = data * time_window * angle_window

    spectrum = np.fft.fft2(windowed)
    spectrum_mag = np.abs(spectrum)

    temporal_freqs = np.fft.fftfreq(windowed.shape[0], d=1.0 / fps)
    spatial_modes = np.fft.fftfreq(windowed.shape[1], d=1.0 / windowed.shape[1])

    temporal_mask = (temporal_freqs >= min_temporal_hz) & (temporal_freqs <= max_temporal_hz)
    spatial_mask = (np.abs(spatial_modes) >= 1) & (np.abs(spatial_modes) <= max_spatial_mode)

    if not np.any(temporal_mask) or not np.any(spatial_mask):
        raise ValueError("可用频率范围为空，请检查参数设置。")

    search_mag = spectrum_mag[np.ix_(temporal_mask, spatial_mask)]
    peak_flat_idx = int(np.argmax(search_mag))
    peak_time_idx, peak_mode_idx = np.unravel_index(peak_flat_idx, search_mag.shape)

    peak_temporal_hz = float(temporal_freqs[temporal_mask][peak_time_idx])
    peak_spatial_mode = int(round(float(spatial_modes[spatial_mask][peak_mode_idx])))
    peak_magnitude = float(search_mag[peak_time_idx, peak_mode_idx])
    rotor_freq_hz = abs(peak_temporal_hz / peak_spatial_mode)
    return (
        rotor_freq_hz,
        peak_temporal_hz,
        peak_spatial_mode,
        peak_magnitude,
        windowed,
        spectrum_mag,
        temporal_freqs,
        spatial_modes,
    )


def resolve_video_path(video_arg: str | None) -> Path:
    if video_arg:
        video_path = Path(video_arg)
        if not video_path.exists():
            raise FileNotFoundError(f"指定视频不存在: {video_path}")
        return video_path

    if not DEFAULT_VIDEO_DIR.exists():
        raise FileNotFoundError(f"默认视频目录不存在: {DEFAULT_VIDEO_DIR}")

    candidates = sorted(DEFAULT_VIDEO_DIR.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"{DEFAULT_VIDEO_DIR} 下没有 .mp4 文件。")

    return candidates[0]


def resolve_output_dir(output_dir_arg: str | None, video_path: Path) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return DEFAULT_OUTPUT_DIR / video_path.stem


def render_summary_figure(
    time_angle_map: np.ndarray,
    debug_edges: np.ndarray | None,
    processed_map: np.ndarray,
    spectrum_mag: np.ndarray,
    temporal_freqs: np.ndarray,
    spatial_modes: np.ndarray,
    frame_idx: int,
    fps: float,
    peak_temporal_hz: float,
    peak_spatial_mode: int,
) -> plt.Figure:
    positive_time = temporal_freqs >= 0
    mode_limit = np.abs(spatial_modes) <= 12

    fig = plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(
        time_angle_map.T,
        aspect="auto",
        cmap="hot",
        extent=[0, frame_idx / fps, 360, 0],
    )
    plt.title("Time-Angle Structural Map")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")

    plt.subplot(2, 2, 2)
    if debug_edges is not None:
        plt.imshow(debug_edges, aspect="auto", cmap="gray")
    plt.title("Polar Edge Map")
    plt.xlabel("Radius bin")
    plt.ylabel("Angle bin")

    plt.subplot(2, 2, 3)
    plt.imshow(
        processed_map.T,
        aspect="auto",
        cmap="coolwarm",
        extent=[0, frame_idx / fps, 360, 0],
    )
    plt.title("Processed Map (k=0 removed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")

    plt.subplot(2, 2, 4)
    plt.imshow(
        np.log1p(spectrum_mag[np.ix_(positive_time, mode_limit)]),
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=[
            spatial_modes[mode_limit][0],
            spatial_modes[mode_limit][-1],
            temporal_freqs[positive_time][0],
            temporal_freqs[positive_time][-1],
        ],
    )
    plt.axvline(peak_spatial_mode, color="cyan", linestyle="--", linewidth=1)
    plt.axhline(peak_temporal_hz, color="cyan", linestyle="--", linewidth=1)
    plt.title("2D Spectrum: Temporal Frequency vs Angular Mode")
    plt.xlabel("Angular mode k (cycles/revolution)")
    plt.ylabel("Temporal frequency (Hz)")

    plt.tight_layout()
    return fig


__all__ = [
    "DEFAULT_ANGULAR_RESOLUTION",
    "build_angle_profile",
    "analyze_spatiotemporal_frequency",
    "resolve_video_path",
    "resolve_output_dir",
    "render_summary_figure",
]
