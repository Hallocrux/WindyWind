from __future__ import annotations

import warnings

import numpy as np
from scipy import signal


def build_harmonic_mask(
    freqs: np.ndarray,
    *,
    rpm: float | None,
    harmonic_orders: tuple[int, ...],
    half_width: float,
) -> tuple[np.ndarray, list[dict[str, float | int | None]]]:
    keep_mask = np.ones_like(freqs, dtype=bool)
    rows: list[dict[str, float | int | None]] = []
    if rpm is None or not np.isfinite(rpm) or rpm <= 0:
        return keep_mask, rows

    rotor_hz = float(rpm) / 60.0
    for harmonic_order in harmonic_orders:
        harmonic_hz = rotor_hz * harmonic_order
        band_start = harmonic_hz - half_width
        band_end = harmonic_hz + half_width
        keep_mask &= (freqs < band_start) | (freqs > band_end)
        rows.append(
            {
                "harmonic_order": int(harmonic_order),
                "rotor_hz": rotor_hz,
                "harmonic_hz": harmonic_hz,
                "band_start_hz": band_start,
                "band_end_hz": band_end,
            }
        )
    return keep_mask, rows


def compute_spectral_matrices(
    matrix: np.ndarray,
    *,
    sampling_rate: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("输入矩阵必须为二维数组。")
    n_samples, n_channels = matrix.shape
    if n_samples < 8:
        raise ValueError("样本点过少，无法计算频谱。")

    nperseg = min(n_samples, nperseg or n_samples)
    noverlap = max(0, nperseg // 2)
    freqs, _ = signal.welch(
        matrix[:, 0],
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    csd_matrix = np.zeros((freqs.size, n_channels, n_channels), dtype=np.complex128)
    coherence_matrix = np.zeros((freqs.size, n_channels, n_channels), dtype=float)

    for i in range(n_channels):
        for j in range(i, n_channels):
            _, pxy = signal.csd(
                matrix[:, i],
                matrix[:, j],
                fs=sampling_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
            csd_matrix[:, i, j] = pxy
            csd_matrix[:, j, i] = np.conj(pxy)
            if i == j:
                coherence_matrix[:, i, j] = 1.0
                continue
            _, cxy = signal.coherence(
                matrix[:, i],
                matrix[:, j],
                fs=sampling_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
            )
            coherence_matrix[:, i, j] = cxy
            coherence_matrix[:, j, i] = cxy
    return freqs, csd_matrix, coherence_matrix


def compute_fdd_spectrum(csd_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_freq, n_channels, _ = csd_matrix.shape
    singular_values = np.zeros((n_freq, n_channels), dtype=float)
    singular_vectors = np.zeros((n_freq, n_channels, n_channels), dtype=np.complex128)
    for index in range(n_freq):
        hermitian = 0.5 * (csd_matrix[index] + csd_matrix[index].conj().T)
        u, s, _ = np.linalg.svd(hermitian, full_matrices=True)
        singular_values[index] = s
        singular_vectors[index] = u
    return singular_values, singular_vectors


def select_peak_index(
    freqs: np.ndarray,
    singular_values: np.ndarray,
    *,
    freq_min: float,
    freq_max: float,
    focus_min: float,
    focus_max: float,
    keep_mask: np.ndarray | None = None,
) -> int | None:
    keep_mask = np.ones_like(freqs, dtype=bool) if keep_mask is None else keep_mask.astype(bool)
    global_mask = keep_mask & (freqs >= freq_min) & (freqs <= freq_max)
    focus_mask = global_mask & (freqs >= focus_min) & (freqs <= focus_max)
    if np.any(focus_mask):
        masked = np.where(focus_mask, singular_values, -np.inf)
        index = int(np.argmax(masked))
        if np.isfinite(masked[index]):
            return index
    if np.any(global_mask):
        masked = np.where(global_mask, singular_values, -np.inf)
        index = int(np.argmax(masked))
        if np.isfinite(masked[index]):
            return index
    return None


def normalize_mode_shape(mode_shape: np.ndarray) -> np.ndarray:
    vector = np.asarray(mode_shape, dtype=np.complex128).reshape(-1)
    if vector.size == 0:
        return np.array([], dtype=float)
    abs_vector = np.abs(vector)
    if np.all(abs_vector < 1e-12):
        return np.zeros(vector.size, dtype=float)

    ref_index = int(np.argmax(abs_vector > 1e-12))
    ref_value = vector[ref_index]
    aligned = vector * np.exp(-1j * np.angle(ref_value))
    real_vector = np.real_if_close(aligned, tol=1000)
    if np.iscomplexobj(real_vector):
        real_vector = np.real(real_vector)
    real_vector = np.asarray(real_vector, dtype=float)
    scale = float(np.max(np.abs(real_vector)))
    if scale <= 0:
        return np.zeros(real_vector.size, dtype=float)
    normalized = real_vector / scale
    first_nonzero = np.flatnonzero(np.abs(normalized) > 1e-12)
    if first_nonzero.size and normalized[first_nonzero[0]] < 0:
        normalized = -normalized
    return normalized


def compute_mac(shape_a: np.ndarray, shape_b: np.ndarray) -> float:
    phi_a = np.asarray(shape_a, dtype=np.complex128).reshape(-1)
    phi_b = np.asarray(shape_b, dtype=np.complex128).reshape(-1)
    denom = np.vdot(phi_a, phi_a) * np.vdot(phi_b, phi_b)
    if abs(denom) <= 1e-12:
        return 0.0
    score = abs(np.vdot(phi_a, phi_b)) ** 2 / abs(denom)
    return float(np.clip(np.real(score), 0.0, 1.0))


def estimate_efdd_damping_ratio(
    freqs: np.ndarray,
    singular_curve: np.ndarray,
    peak_index: int,
) -> float:
    if peak_index <= 0 or peak_index >= len(freqs) - 1:
        return float("nan")
    curve = np.asarray(singular_curve, dtype=float)
    curve = np.maximum(curve, 0.0)
    if curve[peak_index] <= 0:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widths, _, left_ips, right_ips = signal.peak_widths(curve, [peak_index], rel_height=0.5)
    except ValueError:
        return float("nan")
    if widths.size == 0:
        return float("nan")

    left_freq = np.interp(left_ips[0], np.arange(len(freqs), dtype=float), freqs)
    right_freq = np.interp(right_ips[0], np.arange(len(freqs), dtype=float), freqs)
    peak_freq = float(freqs[peak_index])
    if peak_freq <= 0 or right_freq <= left_freq:
        return float("nan")
    damping_ratio = (right_freq - left_freq) / (2.0 * peak_freq)
    if damping_ratio <= 0 or damping_ratio >= 0.5:
        return float("nan")
    return float(damping_ratio)


def summarize_peak_coherence(coherence_matrix: np.ndarray, peak_index: int) -> float:
    slice_2d = coherence_matrix[peak_index]
    triu = slice_2d[np.triu_indices_from(slice_2d, k=1)]
    if triu.size == 0:
        return float("nan")
    return float(np.mean(triu))


def align_mode_shape_series(mode_shapes: list[np.ndarray]) -> np.ndarray:
    if not mode_shapes:
        return np.array([], dtype=float)
    reference = normalize_mode_shape(mode_shapes[0])
    aligned: list[np.ndarray] = [reference]
    for shape in mode_shapes[1:]:
        candidate = normalize_mode_shape(shape)
        if candidate.size != reference.size:
            continue
        if np.dot(reference, candidate) < 0:
            candidate = -candidate
        aligned.append(candidate)
    return normalize_mode_shape(np.median(np.vstack(aligned), axis=0))


def safe_log10(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), 1e-12))


def build_frequency_mask(freqs: np.ndarray, freq_min: float, freq_max: float) -> np.ndarray:
    return (freqs >= freq_min) & (freqs <= freq_max)
