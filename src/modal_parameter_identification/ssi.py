from __future__ import annotations

import numpy as np

from .spectral import compute_mac, normalize_mode_shape


def run_ssi_cov(
    outputs: np.ndarray,
    *,
    sampling_rate: float,
    block_rows: int,
    min_order: int,
    max_order: int,
    freq_min: float,
    freq_max: float,
    max_damping_ratio: float,
) -> list[dict[str, object]]:
    y = np.asarray(outputs, dtype=float)
    if y.ndim != 2:
        raise ValueError("SSI-COV 输入必须为二维矩阵。")
    n_samples, n_outputs = y.shape
    if n_samples <= 2 * block_rows + 2:
        return []

    centered = y - np.mean(y, axis=0, keepdims=True)
    covariances = _compute_covariances(centered, max_lag=2 * block_rows)
    toeplitz = _build_block_toeplitz(covariances, block_rows)
    if toeplitz.size == 0:
        return []

    u, singular_values, _ = np.linalg.svd(toeplitz, full_matrices=False)
    poles: list[dict[str, object]] = []
    max_order = min(max_order, int(np.sum(singular_values > 1e-10)))
    for order in range(min_order, max_order + 1):
        u_r = u[:, :order]
        s_r = singular_values[:order]
        if np.any(s_r <= 1e-12):
            continue
        sqrt_s = np.diag(np.sqrt(s_r))
        observability = u_r @ sqrt_s
        c_matrix = observability[:n_outputs, :]
        up = observability[:-n_outputs, :]
        down = observability[n_outputs:, :]
        if up.size == 0 or down.size == 0:
            continue
        a_matrix, *_ = np.linalg.lstsq(up, down, rcond=None)
        eigvals, eigvecs = np.linalg.eig(a_matrix)
        for pole_index, eigval in enumerate(eigvals):
            if abs(eigval) <= 1e-10 or abs(eigval) >= 1.0:
                continue
            continuous = np.log(complex(eigval)) * sampling_rate
            if np.imag(continuous) <= 0:
                continue
            freq_hz = abs(np.imag(continuous)) / (2.0 * np.pi)
            if not (freq_min <= freq_hz <= freq_max):
                continue
            damping_ratio = -np.real(continuous) / max(abs(continuous), 1e-12)
            if not np.isfinite(damping_ratio) or damping_ratio < 0 or damping_ratio > max_damping_ratio:
                continue
            mode_shape = c_matrix @ eigvecs[:, pole_index]
            poles.append(
                {
                    "order": int(order),
                    "pole_index": int(pole_index),
                    "frequency_hz": float(freq_hz),
                    "damping_ratio": float(damping_ratio),
                    "mode_shape": normalize_mode_shape(mode_shape),
                    "eig_real": float(np.real(eigval)),
                    "eig_imag": float(np.imag(eigval)),
                }
            )
    return poles


def label_stable_poles(
    poles: list[dict[str, object]],
    *,
    freq_tol_hz: float,
    damping_tol: float,
    mac_tol: float,
) -> list[dict[str, object]]:
    by_order: dict[int, list[dict[str, object]]] = {}
    for pole in poles:
        by_order.setdefault(int(pole["order"]), []).append(dict(pole))

    labeled: list[dict[str, object]] = []
    previous_order: list[dict[str, object]] = []
    for order in sorted(by_order):
        current_order = by_order[order]
        for pole in current_order:
            is_stable = False
            best_mac = 0.0
            for previous in previous_order:
                if abs(float(pole["frequency_hz"]) - float(previous["frequency_hz"])) > freq_tol_hz:
                    continue
                if abs(float(pole["damping_ratio"]) - float(previous["damping_ratio"])) > damping_tol:
                    continue
                mac = compute_mac(
                    np.asarray(pole["mode_shape"], dtype=float),
                    np.asarray(previous["mode_shape"], dtype=float),
                )
                if mac >= mac_tol and mac > best_mac:
                    is_stable = True
                    best_mac = mac
            pole["is_stable"] = int(is_stable)
            pole["stable_mac"] = float(best_mac)
            labeled.append(pole)
        previous_order = current_order
    return labeled


def assign_mode_clusters(
    poles: list[dict[str, object]],
    *,
    freq_tol_hz: float,
    mac_tol: float,
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for pole in sorted(poles, key=lambda item: (float(item["frequency_hz"]), int(item["order"]))):
        assigned = False
        for cluster in clusters:
            freq_center = float(np.median([float(item["frequency_hz"]) for item in cluster["items"]]))
            if abs(float(pole["frequency_hz"]) - freq_center) > freq_tol_hz:
                continue
            mac = compute_mac(
                np.asarray(pole["mode_shape"], dtype=float),
                np.asarray(cluster["items"][0]["mode_shape"], dtype=float),
            )
            if mac < mac_tol:
                continue
            cluster["items"].append(pole)
            pole["cluster_id"] = int(cluster["cluster_id"])
            assigned = True
            break
        if not assigned:
            cluster_id = len(clusters)
            pole["cluster_id"] = cluster_id
            clusters.append({"cluster_id": cluster_id, "items": [pole]})
    return poles


def select_dominant_cluster(
    poles: list[dict[str, object]],
    *,
    reference_frequency_hz: float,
    focus_min: float,
    focus_max: float,
) -> dict[str, object] | None:
    if not poles:
        return None
    stable_poles = [pole for pole in poles if int(pole.get("is_stable", 0)) == 1]
    candidate_poles = stable_poles or poles
    clusters: dict[int, list[dict[str, object]]] = {}
    for pole in candidate_poles:
        clusters.setdefault(int(pole.get("cluster_id", -1)), []).append(pole)

    best: dict[str, object] | None = None
    for cluster_id, items in clusters.items():
        freqs = np.array([float(item["frequency_hz"]) for item in items], dtype=float)
        median_freq = float(np.median(freqs))
        if not (focus_min <= median_freq <= focus_max):
            continue
        score = (
            int(sum(int(item.get("is_stable", 0)) for item in items)),
            -abs(median_freq - reference_frequency_hz),
            len(items),
        )
        if best is None or score > best["score"]:
            best = {
                "cluster_id": int(cluster_id),
                "score": score,
                "frequency_hz": median_freq,
                "damping_ratio": float(np.median([float(item["damping_ratio"]) for item in items])),
                "mode_shape": _aggregate_cluster_shape(items),
                "pole_count": len(items),
                "stable_count": int(sum(int(item.get("is_stable", 0)) for item in items)),
            }
    return best


def _compute_covariances(outputs: np.ndarray, max_lag: int) -> list[np.ndarray]:
    n_samples = outputs.shape[0]
    covariances: list[np.ndarray] = []
    for lag in range(max_lag + 1):
        lhs = outputs[lag:]
        rhs = outputs[: n_samples - lag]
        covariances.append((lhs.T @ rhs) / max(len(lhs), 1))
    return covariances


def _build_block_toeplitz(covariances: list[np.ndarray], block_rows: int) -> np.ndarray:
    if len(covariances) < 2 * block_rows:
        return np.array([], dtype=float)
    blocks: list[list[np.ndarray]] = []
    for row in range(block_rows):
        block_row: list[np.ndarray] = []
        for col in range(block_rows):
            block_row.append(covariances[row + col + 1])
        blocks.append(block_row)
    return np.block(blocks)


def _aggregate_cluster_shape(items: list[dict[str, object]]) -> np.ndarray:
    shapes = [np.asarray(item["mode_shape"], dtype=float) for item in items]
    reference = shapes[0]
    aligned: list[np.ndarray] = []
    for shape in shapes:
        candidate = shape.copy()
        if np.dot(reference, candidate) < 0:
            candidate = -candidate
        aligned.append(candidate)
    return normalize_mode_shape(np.median(np.vstack(aligned), axis=0))
