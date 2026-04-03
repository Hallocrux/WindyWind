from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SAMPLE_DIR = Path("outputs/windyWindHowfast/VID_20260330_162635")
DEFAULT_VIDEO_PATH = Path("data/video/VID_20260330_162635.mp4")
DEFAULT_OUTPUT_DIR = Path("outputs/try/002_auto_roi_failure_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量化分析自动 ROI 失败原因。")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--reference-roi", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2

    term1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    term2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    term3 = 0.5 * math.sqrt(
        max(
            0.0,
            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2),
        )
    )
    return term1 + term2 - term3


def circle_iou(a: dict[str, float], b: dict[str, float]) -> float:
    distance = math.hypot(a["center_x"] - b["center_x"], a["center_y"] - b["center_y"])
    intersection = circle_intersection_area(a["radius"], b["radius"], distance)
    union = math.pi * a["radius"] ** 2 + math.pi * b["radius"] ** 2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_first_frame(video_path: Path) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频首帧: {video_path}")
    return frame


def build_candidate_table(
    candidate_records: list[dict[str, Any]],
    reference_roi: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in candidate_records:
        candidate = record["candidate"]
        score = record["score_breakdown"]
        center_distance = math.hypot(
            float(candidate["center_x"]) - reference_roi["center_x"],
            float(candidate["center_y"]) - reference_roi["center_y"],
        )
        radius_abs_error = abs(float(candidate["radius"]) - reference_roi["radius"])
        rows.append(
            {
                "rank": int(record["rank"]) if record["rank"] is not None else None,
                "source": candidate["source"],
                "variant": candidate["metadata"].get("variant"),
                "center_x": float(candidate["center_x"]),
                "center_y": float(candidate["center_y"]),
                "radius": float(candidate["radius"]),
                "total_score": float(score["total_score"]),
                "generator_prior": float(candidate["score"]),
                "center_consistency_score": float(score["center_consistency_score"]),
                "radius_consistency_score": float(score["radius_consistency_score"]),
                "signal_energy_score": float(score["signal_energy_score"]),
                "occlusion_penalty": float(score["occlusion_penalty"]),
                "center_distance_to_reference": center_distance,
                "radius_abs_error_to_reference": radius_abs_error,
                "radius_rel_error_to_reference": radius_abs_error / max(reference_roi["radius"], 1.0),
                "iou_to_reference": circle_iou(
                    {
                        "center_x": float(candidate["center_x"]),
                        "center_y": float(candidate["center_y"]),
                        "radius": float(candidate["radius"]),
                    },
                    reference_roi,
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["rank", "total_score"], ascending=[True, False]).reset_index(drop=True)


def draw_overlay(
    frame_bgr: Any,
    df: pd.DataFrame,
    reference_roi: dict[str, float],
    output_path: Path,
    top_k: int,
) -> None:
    canvas = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    ref_center = (int(round(reference_roi["center_x"])), int(round(reference_roi["center_y"])))
    ref_radius = int(round(reference_roi["radius"]))
    cv2.circle(canvas, ref_center, ref_radius, (0, 255, 255), 4)
    cv2.circle(canvas, ref_center, 5, (0, 255, 255), -1)
    cv2.putText(canvas, "reference", (ref_center[0] + 12, ref_center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    for _, row in df.head(top_k).iterrows():
        center = (int(round(row["center_x"])), int(round(row["center_y"])))
        radius = int(round(row["radius"]))
        rank = int(row["rank"])
        color = (255, 64, 64) if rank == 1 else (80, 255, 80)
        thickness = 4 if rank == 1 else 2
        cv2.circle(canvas, center, radius, color, thickness)
        cv2.circle(canvas, center, 4, color, -1)
        label = f"#{rank} {row['source']}"
        cv2.putText(canvas, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    plt.figure(figsize=(9, 16))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_summary(
    df: pd.DataFrame,
    detection: dict[str, Any],
    reference_roi: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    selected_row = df.iloc[0].to_dict()
    best_iou_row = df.sort_values("iou_to_reference", ascending=False).iloc[0].to_dict()
    nearest_row = df.sort_values("center_distance_to_reference", ascending=True).iloc[0].to_dict()

    summary = {
        "reference_roi": reference_roi,
        "selected_rank": int(selected_row["rank"]),
        "selected_source": selected_row["source"],
        "selected_variant": selected_row["variant"],
        "selected_center_distance_to_reference": float(selected_row["center_distance_to_reference"]),
        "selected_radius_abs_error_to_reference": float(selected_row["radius_abs_error_to_reference"]),
        "selected_iou_to_reference": float(selected_row["iou_to_reference"]),
        "best_iou_rank": int(best_iou_row["rank"]),
        "best_iou_to_reference": float(best_iou_row["iou_to_reference"]),
        "best_iou_center_distance_to_reference": float(best_iou_row["center_distance_to_reference"]),
        "nearest_center_rank": int(nearest_row["rank"]),
        "nearest_center_distance_to_reference": float(nearest_row["center_distance_to_reference"]),
        "candidates_within_120px": int((df["center_distance_to_reference"] <= 120).sum()),
        "candidates_within_180px": int((df["center_distance_to_reference"] <= 180).sum()),
        "candidates_with_iou_ge_0_5": int((df["iou_to_reference"] >= 0.5).sum()),
        "top_score": float(df.iloc[0]["total_score"]),
        "score_gap_to_best_iou": float(df.iloc[0]["total_score"] - best_iou_row["total_score"]),
        "roi_detection_status": detection["status"],
        "roi_reference_frame_indices": detection["reference_frame_indices"],
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    lines = [
        "# 自动 ROI 失败诊断摘要",
        "",
        f"- 参考 ROI: center=({reference_roi['center_x']:.0f}, {reference_roi['center_y']:.0f}), radius={reference_roi['radius']:.0f}",
        f"- 自动选中候选 rank#{summary['selected_rank']} ({summary['selected_source']} / {summary['selected_variant']})",
        f"- 自动选中中心距参考轮毂约 {summary['selected_center_distance_to_reference']:.1f} px，半径误差 {summary['selected_radius_abs_error_to_reference']:.1f} px，IoU={summary['selected_iou_to_reference']:.3f}",
        f"- 与参考 ROI IoU 最高的候选是 rank#{summary['best_iou_rank']}，IoU={summary['best_iou_to_reference']:.3f}",
        f"- 与参考中心最近的候选是 rank#{summary['nearest_center_rank']}，中心误差 {summary['nearest_center_distance_to_reference']:.1f} px",
        f"- 中心落在参考轮毂 120 px 内的候选数: {summary['candidates_within_120px']}",
        f"- 与参考 ROI IoU >= 0.5 的候选数: {summary['candidates_with_iou_ge_0_5']}",
        "",
        "## 直接结论",
        "",
        "- 这次失败首先发生在候选生成阶段：几乎没有候选真正落在轮毂附近。",
        "- 评分阶段也存在偏置：错误候选虽然远离轮毂，但因为局部运动环自洽，所以得到了较高的 center/radius consistency score。",
        "- 当前评分更像是在寻找“稳定的圆形运动区域”，而不是寻找“叶片共同绕转的真实轮毂”。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir
    reference_roi_path = args.reference_roi or sample_dir / "cycle_a_roi.json"

    detection = load_json(sample_dir / "analysis_roi_detection.json")
    candidate_records = load_json(sample_dir / "analysis_roi_candidates.json")
    reference_roi = {
        "center_x": float(load_json(reference_roi_path)["center_x"]),
        "center_y": float(load_json(reference_roi_path)["center_y"]),
        "radius": float(load_json(reference_roi_path)["radius"]),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_candidate_table(candidate_records=candidate_records, reference_roi=reference_roi)
    df.to_csv(output_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")

    frame = load_first_frame(args.video)
    draw_overlay(
        frame_bgr=frame,
        df=df,
        reference_roi=reference_roi,
        output_path=output_dir / "top_candidates_overlay.png",
        top_k=max(1, int(args.top_k)),
    )
    summary = write_summary(df=df, detection=detection, reference_roi=reference_roi, output_dir=output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
