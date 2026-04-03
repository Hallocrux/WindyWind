from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .constants import DEFAULT_ROI_FRAME_STRATEGY, DEFAULT_ROI_REFERENCE_MAX_FRAMES, DEFAULT_ROI_SCORE_THRESHOLD
from .models import (
    ROICandidate,
    ROICandidateScore,
    ROIConfig,
    ROIReferenceContext,
    ROIReferenceFrames,
    ROISelectionResult,
)
from .support import compute_edge_map, downscale_gray, normalize_map, read_frames_at_indices


def load_roi_config(roi_path: Path) -> ROIConfig:
    with roi_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    required = {"center_x", "center_y", "radius"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"ROI 文件缺少字段: {sorted(missing)}")

    return ROIConfig(
        center_x=int(data["center_x"]),
        center_y=int(data["center_y"]),
        radius=int(data["radius"]),
    )


def save_roi_config(roi_config: ROIConfig, roi_path: Path) -> None:
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    with roi_path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(roi_config), fp, ensure_ascii=False, indent=2)


def candidate_from_roi_config(
    roi_config: ROIConfig,
    source: str,
    score: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> ROICandidate:
    return ROICandidate(
        center_x=float(roi_config.center_x),
        center_y=float(roi_config.center_y),
        radius=float(roi_config.radius),
        score=float(score),
        source=source,
        metadata=metadata or {},
    )


def roi_config_from_candidate(candidate: ROICandidate) -> ROIConfig:
    return ROIConfig(
        center_x=int(round(candidate.center_x)),
        center_y=int(round(candidate.center_y)),
        radius=max(1, int(round(candidate.radius))),
    )


def sample_reference_frames(
    cap: cv2.VideoCapture,
    strategy: str = DEFAULT_ROI_FRAME_STRATEGY,
    max_frames: int = DEFAULT_ROI_REFERENCE_MAX_FRAMES,
    manual_index_list: list[int] | None = None,
) -> ROIReferenceFrames:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("视频帧数无效，无法采样参考帧。")

    max_frames = max(1, int(max_frames))
    metadata: dict[str, Any] = {"total_frames": total_frames}

    if strategy == "manual_index_list":
        if not manual_index_list:
            raise ValueError("manual_index_list 策略需要提供 frame index 列表。")
        frame_indices = sorted({int(i) for i in manual_index_list if 0 <= int(i) < total_frames})
        frames = read_frames_at_indices(cap, frame_indices)
        metadata["requested_indices"] = frame_indices
        return ROIReferenceFrames(
            strategy=strategy,
            frame_indices=frame_indices,
            frames=frames,
            metadata=metadata,
        )

    if strategy not in {"early_stable", "uniform_prefix", "low_motion_subset"}:
        raise ValueError(f"不支持的 roi frame strategy: {strategy}")

    if strategy in {"early_stable", "uniform_prefix"}:
        pool_size = min(total_frames, max(max_frames * 6, max_frames))
        pool_indices = np.linspace(0, pool_size - 1, num=min(pool_size, max_frames * 4), dtype=int)
    else:
        pool_size = min(total_frames, max(max_frames * 10, max_frames))
        pool_indices = np.linspace(0, total_frames - 1, num=min(pool_size, max_frames * 4), dtype=int)

    pool_indices = sorted({int(i) for i in pool_indices})
    pool_frames = read_frames_at_indices(cap, pool_indices)
    if not pool_frames:
        raise ValueError("无法读取参考帧。")

    if strategy == "uniform_prefix":
        chosen_indices = pool_indices[:max_frames]
        chosen_frames = pool_frames[:max_frames]
    else:
        downscaled = [downscale_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in pool_frames]
        motion_scores: list[float] = [0.0]
        for prev, current in zip(downscaled, downscaled[1:]):
            diff = cv2.absdiff(current, prev)
            motion_scores.append(float(np.mean(diff)))

        ranked = sorted(
            zip(pool_indices, pool_frames, motion_scores),
            key=lambda item: (item[2], item[0]),
        )
        chosen = ranked[: max_frames]
        chosen.sort(key=lambda item: item[0])
        chosen_indices = [item[0] for item in chosen]
        chosen_frames = [item[1] for item in chosen]
        metadata["pool_motion_scores"] = [
            {"frame_index": idx, "motion_score": score}
            for idx, score in zip(pool_indices, motion_scores)
        ]

    metadata["pool_indices"] = pool_indices
    return ROIReferenceFrames(
        strategy=strategy,
        frame_indices=chosen_indices,
        frames=chosen_frames,
        metadata=metadata,
    )


def build_reference_context(reference_frames: ROIReferenceFrames) -> ROIReferenceContext:
    if not reference_frames.frames:
        raise ValueError("参考帧为空，无法构造 ROI 上下文。")

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in reference_frames.frames]
    edge_frames = [compute_edge_map(gray) for gray in gray_frames]

    motion_maps: list[np.ndarray] = []
    previous_gray = gray_frames[0]
    for gray in gray_frames:
        diff = cv2.absdiff(gray, previous_gray)
        motion_maps.append(normalize_map(cv2.GaussianBlur(diff, (7, 7), 0)))
        previous_gray = gray

    mean_gray = np.mean(np.stack(gray_frames).astype(np.float32), axis=0)
    mean_edges = normalize_map(np.mean(np.stack(edge_frames), axis=0))
    mean_motion = normalize_map(np.mean(np.stack(motion_maps), axis=0))

    combined_frames = [
        normalize_map(0.65 * motion_map + 0.35 * edge_map)
        for motion_map, edge_map in zip(motion_maps, edge_frames)
    ]
    combined_map = normalize_map(0.65 * mean_motion + 0.35 * mean_edges)

    height, width = gray_frames[0].shape[:2]
    return ROIReferenceContext(
        frames=reference_frames.frames,
        frame_indices=reference_frames.frame_indices,
        gray_frames=gray_frames,
        edge_frames=edge_frames,
        motion_maps=motion_maps,
        combined_frames=combined_frames,
        mean_gray=mean_gray,
        mean_edges=mean_edges,
        mean_motion=mean_motion,
        combined_map=combined_map,
        height=height,
        width=width,
    )


def select_roi(event: int, x: int, y: int, flags: int, param: dict) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    scale = param.get("scale", 1.0)
    orig_x = int(x / scale)
    orig_y = int(y / scale)
    if not param.get("center_selected", False):
        param["center"] = (orig_x, orig_y)
        param["center_selected"] = True
        print(f"已选择中心点: {orig_x}, {orig_y}")
        return

    param["outer_point"] = (orig_x, orig_y)
    param["roi_selected"] = True
    print(f"已选择外缘点: {orig_x}, {orig_y}")


def select_roi_interactively(first_frame: np.ndarray) -> ROIConfig | None:
    frame_height = first_frame.shape[0]
    scale = 720 / frame_height if frame_height > 720 else 1.0
    params = {"center_selected": False, "roi_selected": False, "scale": scale}
    window_name = "Select Windmill Area"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_roi, params)

    print("请先点击风机中心，再点击叶尖或叶轮外缘。按 Esc 可取消。")

    while not params["roi_selected"]:
        display = cv2.resize(first_frame, None, fx=scale, fy=scale)
        if params["center_selected"]:
            cx = int(params["center"][0] * scale)
            cy = int(params["center"][1] * scale)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    if not params.get("roi_selected"):
        return None

    center = params["center"]
    outer_point = params["outer_point"]
    radius = int(math.hypot(outer_point[0] - center[0], outer_point[1] - center[1]))
    return ROIConfig(center_x=center[0], center_y=center[1], radius=radius)


def generate_candidates_from_manual(
    first_frame: np.ndarray,
    interactive: bool = True,
) -> list[ROICandidate]:
    if not interactive:
        return []
    roi_config = select_roi_interactively(first_frame)
    if roi_config is None:
        return []
    return [
        candidate_from_roi_config(
            roi_config=roi_config,
            source="manual",
            score=1.0,
            metadata={"generator": "manual"},
        )
    ]


def generate_candidates_from_detector(frames: list[np.ndarray]) -> list[ROICandidate]:
    _ = frames
    return []


def clip_candidate_radius(candidate: ROICandidate, width: int, height: int) -> ROICandidate | None:
    cx = float(candidate.center_x)
    cy = float(candidate.center_y)
    radius = float(candidate.radius)
    if radius <= 4:
        return None
    max_allowed = min(cx, cy, width - 1 - cx, height - 1 - cy)
    if max_allowed <= 4:
        return None
    radius = min(radius, max_allowed)
    if radius <= 4:
        return None
    return ROICandidate(
        center_x=cx,
        center_y=cy,
        radius=radius,
        score=float(candidate.score),
        source=candidate.source,
        metadata=dict(candidate.metadata),
    )


def deduplicate_candidates(
    candidates: list[ROICandidate],
    width: int,
    height: int,
    center_tol_ratio: float = 0.08,
    radius_tol_ratio: float = 0.12,
) -> list[ROICandidate]:
    deduped: list[ROICandidate] = []
    diagonal = math.hypot(width, height)
    center_tol = max(8.0, diagonal * center_tol_ratio)

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        clipped = clip_candidate_radius(candidate, width=width, height=height)
        if clipped is None:
            continue
        duplicate = False
        for existing in deduped:
            center_dist = math.hypot(clipped.center_x - existing.center_x, clipped.center_y - existing.center_y)
            radius_dist = abs(clipped.radius - existing.radius)
            radius_tol = max(clipped.radius, existing.radius) * radius_tol_ratio
            if center_dist <= center_tol and radius_dist <= radius_tol:
                duplicate = True
                break
        if not duplicate:
            deduped.append(clipped)
    return deduped


def estimate_radius_from_support(
    signal_map: np.ndarray,
    center_x: float,
    center_y: float,
    min_radius: float,
    max_radius: float,
    angle_count: int = 48,
    radius_samples: int = 96,
) -> tuple[float | None, dict[str, Any]]:
    height, width = signal_map.shape[:2]
    min_radius = max(4.0, float(min_radius))
    max_radius = min(
        float(max_radius),
        center_x,
        center_y,
        width - 1 - center_x,
        height - 1 - center_y,
    )
    if max_radius <= min_radius:
        return None, {"reason": "invalid_radius_range"}

    radii = np.linspace(min_radius, max_radius, radius_samples, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, angle_count, endpoint=False, dtype=np.float32)
    peak_radii: list[float] = []
    peak_strengths: list[float] = []

    for angle in angles:
        xs = np.clip(np.round(center_x + np.cos(angle) * radii).astype(int), 0, width - 1)
        ys = np.clip(np.round(center_y + np.sin(angle) * radii).astype(int), 0, height - 1)
        values = signal_map[ys, xs]
        peak_idx = int(np.argmax(values))
        peak_value = float(values[peak_idx])
        if peak_value > 0.08:
            peak_radii.append(float(radii[peak_idx]))
            peak_strengths.append(peak_value)

    if not peak_radii:
        return None, {"reason": "no_supported_radius"}

    return float(np.median(peak_radii)), {
        "supported_angles": len(peak_radii),
        "peak_strength_mean": float(np.mean(peak_strengths)),
        "peak_radius_std": float(np.std(peak_radii)),
    }


def generate_candidates_from_motion(context: ROIReferenceContext) -> list[ROICandidate]:
    motion_map = context.mean_motion
    if float(np.max(motion_map)) < 0.05:
        return []

    active_values = motion_map[motion_map > 0]
    if active_values.size == 0:
        return []

    threshold = float(np.percentile(active_values, 85))
    mask = (motion_map >= threshold).astype(np.uint8) * 255
    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[ROICandidate] = []
    frame_area = float(context.width * context.height)

    for contour_index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:6]):
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.002:
            continue

        (center_x, center_y), contour_radius = cv2.minEnclosingCircle(contour)
        if contour_radius < 12:
            continue

        estimated_radius, radius_meta = estimate_radius_from_support(
            signal_map=motion_map,
            center_x=center_x,
            center_y=center_y,
            min_radius=max(12.0, contour_radius * 0.6),
            max_radius=min(context.width, context.height) * 0.48,
        )

        base_score = min(1.0, area / (frame_area * 0.18))
        candidates.append(
            ROICandidate(
                center_x=float(center_x),
                center_y=float(center_y),
                radius=float(contour_radius),
                score=base_score,
                source="motion",
                metadata={
                    "generator": "motion",
                    "contour_index": contour_index,
                    "contour_area": area,
                    "threshold": threshold,
                    "variant": "min_enclosing_circle",
                },
            )
        )

        if estimated_radius is not None:
            candidates.append(
                ROICandidate(
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=float(estimated_radius),
                    score=min(1.0, base_score + 0.08),
                    source="motion",
                    metadata={
                        "generator": "motion",
                        "contour_index": contour_index,
                        "contour_area": area,
                        "threshold": threshold,
                        "variant": "support_refined_radius",
                        "radius_support": radius_meta,
                    },
                )
            )

    ys, xs = np.where(mask > 0)
    if xs.size > 0:
        weights = motion_map[ys, xs]
        center_x = float(np.average(xs, weights=weights))
        center_y = float(np.average(ys, weights=weights))
        estimated_radius, radius_meta = estimate_radius_from_support(
            signal_map=motion_map,
            center_x=center_x,
            center_y=center_y,
            min_radius=16.0,
            max_radius=min(context.width, context.height) * 0.48,
        )
        if estimated_radius is not None:
            candidates.append(
                ROICandidate(
                    center_x=center_x,
                    center_y=center_y,
                    radius=float(estimated_radius),
                    score=0.45,
                    source="motion",
                    metadata={
                        "generator": "motion",
                        "variant": "weighted_centroid",
                        "active_pixel_count": int(xs.size),
                        "radius_support": radius_meta,
                    },
                )
            )

    return deduplicate_candidates(candidates, width=context.width, height=context.height)


def generate_candidates_from_static_structure(context: ROIReferenceContext) -> list[ROICandidate]:
    gray_u8 = np.clip(context.mean_gray, 0, 255).astype(np.uint8)
    edges_u8 = (context.mean_edges * 255.0).astype(np.uint8)

    candidates: list[ROICandidate] = []
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray_u8, (9, 9), 2),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(context.width, context.height) // 10),
        param1=120,
        param2=18,
        minRadius=max(10, min(context.width, context.height) // 80),
        maxRadius=max(20, min(context.width, context.height) // 4),
    )

    if circles is not None:
        for circle_index, circle in enumerate(np.round(circles[0, :]).astype(int)[:6]):
            center_x, center_y, detected_radius = map(int, circle)
            estimated_radius, radius_meta = estimate_radius_from_support(
                signal_map=context.mean_edges,
                center_x=float(center_x),
                center_y=float(center_y),
                min_radius=max(12.0, detected_radius * 1.5),
                max_radius=min(context.width, context.height) * 0.48,
            )
            radius = float(estimated_radius if estimated_radius is not None else detected_radius * 2.5)
            candidates.append(
                ROICandidate(
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=radius,
                    score=0.40,
                    source="static_structure",
                    metadata={
                        "generator": "static_structure",
                        "variant": "hough_circle",
                        "circle_index": circle_index,
                        "detected_radius": detected_radius,
                        "radius_support": radius_meta,
                    },
                )
            )

    contours, _ = cv2.findContours(edges_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(context.width * context.height)
    for contour_index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:8]):
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.0005:
            continue
        (center_x, center_y), contour_radius = cv2.minEnclosingCircle(contour)
        if contour_radius < 8:
            continue
        estimated_radius, radius_meta = estimate_radius_from_support(
            signal_map=context.mean_edges,
            center_x=center_x,
            center_y=center_y,
            min_radius=max(12.0, contour_radius * 1.2),
            max_radius=min(context.width, context.height) * 0.48,
        )
        radius = float(estimated_radius if estimated_radius is not None else contour_radius * 2.0)
        candidates.append(
            ROICandidate(
                center_x=float(center_x),
                center_y=float(center_y),
                radius=radius,
                score=min(0.55, area / (frame_area * 0.03)),
                source="static_structure",
                metadata={
                    "generator": "static_structure",
                    "variant": "edge_contour",
                    "contour_index": contour_index,
                    "contour_area": area,
                    "radius_support": radius_meta,
                },
            )
        )

    return deduplicate_candidates(candidates, width=context.width, height=context.height)


def local_centroid(signal_map: np.ndarray, center_x: float, center_y: float, radius: float) -> tuple[float, float] | None:
    half_size = max(12, int(round(radius * 0.28)))
    height, width = signal_map.shape[:2]
    x0 = max(0, int(round(center_x)) - half_size)
    x1 = min(width, int(round(center_x)) + half_size + 1)
    y0 = max(0, int(round(center_y)) - half_size)
    y1 = min(height, int(round(center_y)) + half_size + 1)
    patch = signal_map[y0:y1, x0:x1]
    if patch.size == 0 or float(np.sum(patch)) <= 1e-6:
        return None

    ys, xs = np.mgrid[y0:y1, x0:x1]
    weight_sum = float(np.sum(patch))
    centroid_x = float(np.sum(xs * patch) / weight_sum)
    centroid_y = float(np.sum(ys * patch) / weight_sum)
    return centroid_x, centroid_y


def angular_ring_coverage(
    signal_map: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    inner_ratio: float = 0.78,
    outer_ratio: float = 1.10,
    angle_bins: int = 24,
) -> dict[str, Any]:
    height, width = signal_map.shape[:2]
    max_radius = int(math.ceil(radius * outer_ratio))
    x0 = max(0, int(round(center_x - max_radius)))
    x1 = min(width, int(round(center_x + max_radius)) + 1)
    y0 = max(0, int(round(center_y - max_radius)))
    y1 = min(height, int(round(center_y + max_radius)) + 1)

    if x0 >= x1 or y0 >= y1:
        return {"coverage": 0.0, "bin_energies": [0.0] * angle_bins}

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs - center_x
    dy = ys - center_y
    distances = np.sqrt(dx * dx + dy * dy)
    annulus = (distances >= radius * inner_ratio) & (distances <= radius * outer_ratio)

    if not np.any(annulus):
        return {"coverage": 0.0, "bin_energies": [0.0] * angle_bins}

    angles = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    bin_indices = np.floor(angles / (2 * np.pi / angle_bins)).astype(int)

    bin_energies = np.zeros(angle_bins, dtype=np.float32)
    annulus_values = signal_map[y0:y1, x0:x1]
    for bin_index in range(angle_bins):
        bin_mask = annulus & (bin_indices == bin_index)
        if np.any(bin_mask):
            bin_energies[bin_index] = float(np.mean(annulus_values[bin_mask]))

    max_bin = float(np.max(bin_energies))
    if max_bin <= 1e-6:
        coverage = 0.0
    else:
        coverage = float(np.mean(bin_energies >= max_bin * 0.25))
    return {"coverage": coverage, "bin_energies": bin_energies.tolist()}


def annulus_mean(
    signal_map: np.ndarray,
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
) -> float:
    height, width = signal_map.shape[:2]
    max_radius = int(math.ceil(outer_radius))
    x0 = max(0, int(round(center_x - max_radius)))
    x1 = min(width, int(round(center_x + max_radius)) + 1)
    y0 = max(0, int(round(center_y - max_radius)))
    y1 = min(height, int(round(center_y + max_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0

    ys, xs = np.mgrid[y0:y1, x0:x1]
    distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    annulus = (distances >= inner_radius) & (distances <= outer_radius)
    if not np.any(annulus):
        return 0.0
    values = signal_map[y0:y1, x0:x1][annulus]
    return float(np.mean(values))


def score_roi_candidate(context: ROIReferenceContext, candidate: ROICandidate) -> ROICandidateScore:
    radius = max(1.0, float(candidate.radius))
    center_x = float(candidate.center_x)
    center_y = float(candidate.center_y)

    centroids: list[tuple[float, float]] = []
    for combined_frame in context.combined_frames:
        centroid = local_centroid(combined_frame, center_x=center_x, center_y=center_y, radius=radius)
        if centroid is not None:
            centroids.append(centroid)

    if centroids:
        distances = [math.hypot(cx - center_x, cy - center_y) / radius for cx, cy in centroids]
        centroid_xs = [item[0] for item in centroids]
        centroid_ys = [item[1] for item in centroids]
        centroid_spread = math.hypot(float(np.std(centroid_xs)), float(np.std(centroid_ys))) / radius
        center_consistency_score = max(
            0.0,
            1.0 - min(1.0, float(np.mean(distances)) * 1.4 + centroid_spread * 1.2),
        )
    else:
        center_consistency_score = 0.0
        centroid_spread = 1.0

    radius_estimates: list[float] = []
    radius_support_details: list[dict[str, Any]] = []
    max_search_radius = min(context.width, context.height) * 0.48
    for combined_frame in context.combined_frames:
        estimated_radius, support_meta = estimate_radius_from_support(
            signal_map=combined_frame,
            center_x=center_x,
            center_y=center_y,
            min_radius=max(8.0, radius * 0.6),
            max_radius=min(max_search_radius, radius * 1.4),
            angle_count=36,
            radius_samples=72,
        )
        if estimated_radius is not None:
            radius_estimates.append(float(estimated_radius))
            radius_support_details.append(support_meta)

    if radius_estimates:
        median_radius = float(np.median(radius_estimates))
        median_error = abs(median_radius - radius) / radius
        radius_spread = float(np.median(np.abs(np.asarray(radius_estimates) - median_radius))) / radius
        valid_ratio = len(radius_estimates) / max(1, len(context.combined_frames))
        radius_consistency_score = max(
            0.0,
            valid_ratio * (1.0 - min(1.0, median_error * 2.0 + radius_spread * 2.0)),
        )
    else:
        median_radius = None
        radius_spread = 1.0
        radius_consistency_score = 0.0

    ring_mean = annulus_mean(context.combined_map, center_x, center_y, radius * 0.78, radius * 1.10)
    inner_mean = annulus_mean(context.combined_map, center_x, center_y, 0.0, radius * 0.35)
    outer_mean = annulus_mean(context.combined_map, center_x, center_y, radius * 1.15, radius * 1.45)
    contrast = max(0.0, ring_mean - 0.5 * (inner_mean + outer_mean))
    signal_energy_score = max(0.0, min(1.0, ring_mean * 1.5 + contrast * 1.8))

    boundary_overflow = (
        max(0.0, radius - center_x)
        + max(0.0, radius - center_y)
        + max(0.0, center_x + radius - (context.width - 1))
        + max(0.0, center_y + radius - (context.height - 1))
    )
    boundary_penalty = min(1.0, boundary_overflow / max(radius * 4.0, 1.0))

    coverage_info = angular_ring_coverage(context.combined_map, center_x, center_y, radius)
    occlusion_penalty = max(0.0, 1.0 - float(coverage_info["coverage"]))

    generator_prior = max(0.0, min(1.0, float(candidate.score)))
    total_score = (
        0.10 * generator_prior
        + 0.30 * center_consistency_score
        + 0.25 * radius_consistency_score
        + 0.35 * signal_energy_score
        - 0.15 * boundary_penalty
        - 0.10 * occlusion_penalty
    )
    total_score = max(0.0, min(1.0, total_score))

    return ROICandidateScore(
        total_score=total_score,
        center_consistency_score=center_consistency_score,
        radius_consistency_score=radius_consistency_score,
        signal_energy_score=signal_energy_score,
        boundary_penalty=boundary_penalty,
        occlusion_penalty=occlusion_penalty,
        metadata={
            "generator_prior": generator_prior,
            "centroid_spread": centroid_spread,
            "radius_median_estimate": median_radius,
            "radius_spread": radius_spread,
            "ring_mean": ring_mean,
            "inner_mean": inner_mean,
            "outer_mean": outer_mean,
            "ring_contrast": contrast,
            "angular_coverage": coverage_info["coverage"],
            "angular_bin_energies": coverage_info["bin_energies"],
            "radius_support_details": radius_support_details,
        },
    )


def candidate_record(
    candidate: ROICandidate,
    candidate_score: ROICandidateScore,
    rank: int | None = None,
) -> dict[str, Any]:
    from .support import jsonable

    return {
        "rank": rank,
        "candidate": jsonable(candidate),
        "score_breakdown": jsonable(candidate_score),
    }


def build_provided_selection_result(
    candidate: ROICandidate,
    status: str,
    source: str,
    reference_frames: ROIReferenceFrames | None = None,
    fallback_used: bool = False,
    failure_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ROISelectionResult:
    selected_score = ROICandidateScore(
        total_score=float(candidate.score),
        center_consistency_score=1.0,
        radius_consistency_score=1.0,
        signal_energy_score=1.0,
        boundary_penalty=0.0,
        occlusion_penalty=0.0,
        metadata={"selection_reason": source},
    )
    frame_strategy = reference_frames.strategy if reference_frames is not None else None
    frame_indices = reference_frames.frame_indices if reference_frames is not None else []
    return ROISelectionResult(
        status=status,
        source=source,
        selected_candidate=candidate,
        selected_score=selected_score,
        candidate_records=[candidate_record(candidate, selected_score, rank=1)],
        failure_reason=failure_reason,
        fallback_used=fallback_used,
        reference_frame_strategy=frame_strategy,
        reference_frame_indices=frame_indices,
        metadata=metadata or {},
    )


def select_best_roi_candidate(
    context: ROIReferenceContext,
    candidates: list[ROICandidate],
    threshold: float,
    reference_frames: ROIReferenceFrames,
) -> ROISelectionResult:
    if not candidates:
        return ROISelectionResult(
            status="failed",
            source="auto",
            selected_candidate=None,
            selected_score=None,
            candidate_records=[],
            failure_reason="no_candidates_generated",
            reference_frame_strategy=reference_frames.strategy,
            reference_frame_indices=reference_frames.frame_indices,
            metadata={"generator_sources": []},
        )

    scored_records: list[tuple[ROICandidate, ROICandidateScore]] = []
    for candidate in candidates:
        scored_records.append((candidate, score_roi_candidate(context, candidate)))

    scored_records.sort(key=lambda item: item[1].total_score, reverse=True)
    records = [
        candidate_record(candidate, score, rank=index + 1)
        for index, (candidate, score) in enumerate(scored_records)
    ]

    best_candidate, best_score = scored_records[0]
    if best_score.total_score < threshold:
        return ROISelectionResult(
            status="failed",
            source="auto",
            selected_candidate=best_candidate,
            selected_score=best_score,
            candidate_records=records,
            failure_reason=f"best_candidate_score_below_threshold:{best_score.total_score:.3f}<{threshold:.3f}",
            reference_frame_strategy=reference_frames.strategy,
            reference_frame_indices=reference_frames.frame_indices,
            metadata={
                "generator_sources": sorted({candidate.source for candidate, _ in scored_records}),
                "score_threshold": threshold,
            },
        )

    return ROISelectionResult(
        status="auto_selected",
        source="auto",
        selected_candidate=best_candidate,
        selected_score=best_score,
        candidate_records=records,
        reference_frame_strategy=reference_frames.strategy,
        reference_frame_indices=reference_frames.frame_indices,
        metadata={
            "generator_sources": sorted({candidate.source for candidate, _ in scored_records}),
            "score_threshold": threshold,
        },
    )


def merge_manual_fallback(
    auto_result: ROISelectionResult,
    manual_candidate: ROICandidate,
) -> ROISelectionResult:
    manual_score = ROICandidateScore(
        total_score=1.0,
        center_consistency_score=1.0,
        radius_consistency_score=1.0,
        signal_energy_score=1.0,
        boundary_penalty=0.0,
        occlusion_penalty=0.0,
        metadata={"selection_reason": "manual_fallback"},
    )
    candidate_records = list(auto_result.candidate_records)
    candidate_records.append(candidate_record(manual_candidate, manual_score, rank=None))
    return ROISelectionResult(
        status="manual_fallback",
        source="manual",
        selected_candidate=manual_candidate,
        selected_score=manual_score,
        candidate_records=candidate_records,
        failure_reason=auto_result.failure_reason,
        fallback_used=True,
        reference_frame_strategy=auto_result.reference_frame_strategy,
        reference_frame_indices=auto_result.reference_frame_indices,
        metadata={
            **auto_result.metadata,
            "fallback_from": auto_result.source,
            "auto_failure_reason": auto_result.failure_reason,
        },
    )


def resolve_roi_candidate(
    cap: cv2.VideoCapture,
    first_frame: np.ndarray,
    *,
    center_x: int | None,
    center_y: int | None,
    radius: int | None,
    roi_path: str | None,
    auto_roi: bool,
    interactive: bool,
    roi_frame_strategy: str = DEFAULT_ROI_FRAME_STRATEGY,
    roi_reference_max_frames: int = DEFAULT_ROI_REFERENCE_MAX_FRAMES,
    roi_score_threshold: float = DEFAULT_ROI_SCORE_THRESHOLD,
) -> tuple[ROIConfig | None, ROISelectionResult]:
    if roi_path:
        roi_config = load_roi_config(Path(roi_path))
        candidate = candidate_from_roi_config(
            roi_config=roi_config,
            source="json",
            score=1.0,
            metadata={"roi_json_path": roi_path},
        )
        return roi_config, build_provided_selection_result(
            candidate=candidate,
            status="provided_json",
            source="json",
        )

    if center_x is not None or center_y is not None or radius is not None:
        if center_x is None or center_y is None or radius is None:
            raise ValueError("使用命令行指定 ROI 时，必须同时提供 --center-x、--center-y、--radius。")
        roi_config = ROIConfig(center_x=center_x, center_y=center_y, radius=radius)
        candidate = candidate_from_roi_config(
            roi_config=roi_config,
            source="cli",
            score=1.0,
            metadata={"provided_via": "cli"},
        )
        return roi_config, build_provided_selection_result(
            candidate=candidate,
            status="provided_cli",
            source="cli",
        )

    auto_result: ROISelectionResult | None = None
    if auto_roi:
        reference_frames = sample_reference_frames(
            cap=cap,
            strategy=roi_frame_strategy,
            max_frames=roi_reference_max_frames,
        )
        context = build_reference_context(reference_frames)
        candidates: list[ROICandidate] = []
        candidates.extend(generate_candidates_from_motion(context))
        candidates.extend(generate_candidates_from_static_structure(context))
        candidates.extend(generate_candidates_from_detector(reference_frames.frames))
        auto_result = select_best_roi_candidate(
            context=context,
            candidates=candidates,
            threshold=roi_score_threshold,
            reference_frames=reference_frames,
        )
        if auto_result.selected_candidate is not None and auto_result.status == "auto_selected":
            return roi_config_from_candidate(auto_result.selected_candidate), auto_result

    manual_candidates = generate_candidates_from_manual(first_frame=first_frame, interactive=interactive)
    if manual_candidates:
        manual_candidate = manual_candidates[0]
        if auto_result is not None:
            manual_result = merge_manual_fallback(auto_result=auto_result, manual_candidate=manual_candidate)
            return roi_config_from_candidate(manual_candidate), manual_result
        return roi_config_from_candidate(manual_candidate), build_provided_selection_result(
            candidate=manual_candidate,
            status="manual_selected",
            source="manual",
        )

    if auto_result is not None:
        return None, auto_result

    return None, ROISelectionResult(
        status="failed",
        source="manual",
        selected_candidate=None,
        selected_score=None,
        candidate_records=[],
        failure_reason="manual_selection_unavailable",
        metadata={"interactive": interactive},
    )
