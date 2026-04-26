from __future__ import annotations

from pathlib import Path
import warnings

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spectral import normalize_mode_shape

POINT_COLUMNS = [f"point_{index}" for index in range(1, 6)]
POINT_POSITIONS = np.arange(1, 6, dtype=float)


def build_mode_shape_animation_frames(
    mode_shape: np.ndarray,
    *,
    cycles: int = 2,
    fps: int = 15,
) -> np.ndarray:
    if cycles <= 0:
        raise ValueError("cycles 必须为正整数。")
    if fps <= 0:
        raise ValueError("fps 必须为正整数。")
    normalized_shape = _validate_mode_shape(mode_shape)
    frame_count = max(int(cycles * fps), fps)
    phases = np.linspace(0.0, 2.0 * np.pi * cycles, frame_count, endpoint=False)
    return np.outer(np.sin(phases), normalized_shape)


def save_mode_shape_animation(
    *,
    case_id: int,
    basis_name: str,
    mode_shape: np.ndarray,
    output_path: Path,
    fps: int = 15,
    cycles: int = 2,
) -> Path:
    normalized_shape = _validate_mode_shape(mode_shape)
    frames = build_mode_shape_animation_frames(normalized_shape, cycles=cycles, fps=fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 7.0), constrained_layout=True)
    baseline_line, = ax.plot(np.zeros_like(POINT_POSITIONS), POINT_POSITIONS, linestyle="--", color="0.65", label="baseline")
    animated_line, = ax.plot(frames[0], POINT_POSITIONS, marker="o", color="tab:blue", linewidth=2.0, label="mode shape")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(POINT_POSITIONS.min() - 0.25, POINT_POSITIONS.max() + 0.25)
    ax.set_xlabel("Normalized lateral displacement")
    ax.set_ylabel("Point index")
    ax.set_title(f"Case {case_id:02d} {basis_name} mode shape")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    def update(frame_index: int) -> tuple[object, object]:
        animated_line.set_xdata(frames[frame_index])
        baseline_line.set_xdata(np.zeros_like(POINT_POSITIONS))
        return animated_line, baseline_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=True,
    )
    try:
        actual_output_path = _save_animation_with_fallback(ani=ani, output_path=output_path, fps=fps)
    finally:
        plt.close(fig)
    return actual_output_path


def save_mode_shape_animations(
    *,
    output_dir: Path,
    shape_tables: dict[str, pd.DataFrame],
    animation_format: str = "auto",
    fps: int = 15,
    cycles: int = 2,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    for basis_name, shape_df in shape_tables.items():
        if shape_df.empty:
            rows.append(
                {
                    "case_id": None,
                    "basis": basis_name,
                    "status": "skipped",
                    "message": "shape table is empty",
                    "path": None,
                }
            )
            continue
        for _, row in shape_df.iterrows():
            case_id = int(row["case_id"])
            result = {
                "case_id": case_id,
                "basis": basis_name,
                "status": "saved",
                "message": "ok",
                "path": None,
            }
            try:
                mode_shape = _extract_mode_shape(row)
            except ValueError as exc:
                result["status"] = "skipped"
                result["message"] = str(exc)
                rows.append(result)
                continue

            output_path = output_dir / _build_animation_file_name(
                case_id=case_id,
                basis_name=basis_name,
                animation_format=animation_format,
            )
            saved_path = save_mode_shape_animation(
                case_id=case_id,
                basis_name=basis_name,
                mode_shape=mode_shape,
                output_path=output_path,
                fps=fps,
                cycles=cycles,
            )
            result["path"] = str(saved_path)
            rows.append(result)
    return rows


def _save_animation_with_fallback(
    *,
    ani: animation.FuncAnimation,
    output_path: Path,
    fps: int,
) -> Path:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        ani.save(output_path, writer=animation.PillowWriter(fps=fps))
        return output_path
    if suffix == ".mp4":
        try:
            ani.save(output_path, writer=animation.FFMpegWriter(fps=fps))
            return output_path
        except Exception as exc:  # pragma: no cover - fallback branch depends on local ffmpeg state
            fallback_path = output_path.with_suffix(".gif")
            warnings.warn(
                f"MP4 写出失败，已回退为 GIF: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            ani.save(fallback_path, writer=animation.PillowWriter(fps=fps))
            return fallback_path
    raise ValueError(f"不支持的动画后缀: {suffix}")


def _extract_mode_shape(row: pd.Series) -> np.ndarray:
    valid_window_count = int(row.get("valid_window_count", 0) or 0)
    if valid_window_count <= 0:
        raise ValueError("no valid windows")
    values = np.array([row.get(column, np.nan) for column in POINT_COLUMNS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("mode shape contains NaN")
    if np.allclose(values, 0.0):
        raise ValueError("mode shape is all zeros")
    return normalize_mode_shape(values)


def _validate_mode_shape(mode_shape: np.ndarray) -> np.ndarray:
    values = np.asarray(mode_shape, dtype=float).reshape(-1)
    if values.size != len(POINT_COLUMNS):
        raise ValueError(f"mode_shape 点数应为 {len(POINT_COLUMNS)}，当前为 {values.size}。")
    if not np.isfinite(values).all():
        raise ValueError("mode_shape 存在非有限值。")
    if np.allclose(values, 0.0):
        raise ValueError("mode_shape 全为 0，无法生成动画。")
    return normalize_mode_shape(values)


def _build_animation_file_name(
    *,
    case_id: int,
    basis_name: str,
    animation_format: str,
) -> str:
    suffix = ".mp4" if animation_format == "auto" else f".{animation_format}"
    return f"case_{case_id:02d}_{basis_name}_mode_shape_animation{suffix}"
