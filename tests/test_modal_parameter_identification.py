from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from src.current.data_loading import DatasetRecord
from src.modal_parameter_identification import animation, fe, io_utils, pipeline, spectral, ssi
from src.modal_parameter_identification import __main__ as modal_cli
from src.modal_parameter_identification import models


class ModalParameterIdentificationTests(unittest.TestCase):
    def test_harmonic_mask_and_sync_rpm_resolution_work(self) -> None:
        freqs = np.linspace(0.0, 5.0, 501)
        keep_mask, rows = spectral.build_harmonic_mask(
            freqs,
            rpm=72.0,
            harmonic_orders=(1, 3),
            half_width=0.1,
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(keep_mask[np.argmin(np.abs(freqs - 1.2))])
        self.assertFalse(keep_mask[np.argmin(np.abs(freqs - 3.6))])

        record = DatasetRecord(
            case_id=7,
            display_name="测试工况",
            file_name="工况7.csv",
            file_path=Path("dummy.csv"),
            wind_speed=4.2,
            rpm=155.0,
            is_labeled=True,
            original_file_name="dummy.csv",
            label_source="unit-test",
            notes="",
        )
        rpm_df = pd.DataFrame(
            {
                "case_id": [7, 7, 8],
                "time": pd.to_datetime(
                    [
                        "2026-04-08 10:00:00",
                        "2026-04-08 10:00:10",
                        "2026-04-08 10:00:10",
                    ]
                ),
                "rpm": [150.0, 156.0, 999.0],
            }
        )
        rpm = io_utils.resolve_window_rpm(
            record=record,
            start_time=pd.Timestamp("2026-04-08 10:00:00"),
            end_time=pd.Timestamp("2026-04-08 10:00:15"),
            rpm_source="sync_csv",
            sync_rpm_df=rpm_df,
        )
        self.assertAlmostEqual(rpm, 153.0, places=6)

    def test_spectral_outputs_have_expected_shapes(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(512, 5))
        freqs, csd_matrix, coherence_matrix = spectral.compute_spectral_matrices(
            matrix,
            sampling_rate=50.0,
            nperseg=256,
        )
        singular_values, singular_vectors = spectral.compute_fdd_spectrum(csd_matrix)

        self.assertEqual(freqs.ndim, 1)
        self.assertEqual(csd_matrix.shape, (freqs.size, 5, 5))
        self.assertEqual(coherence_matrix.shape, (freqs.size, 5, 5))
        self.assertEqual(singular_values.shape, (freqs.size, 5))
        self.assertEqual(singular_vectors.shape, (freqs.size, 5, 5))

    def test_normalize_mode_shape_enforces_scale_and_sign(self) -> None:
        normalized = spectral.normalize_mode_shape(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=6)
        self.assertGreater(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[-1], -1.0, places=6)

    def test_load_fe_reference_validates_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fe.csv"
            pd.DataFrame(
                [
                    {
                        "basis": "strain",
                        "mode_label": "tower_bending_1",
                        "frequency_hz": 2.35,
                        "damping_ratio": 0.02,
                        "point_1": -1.0,
                        "point_2": -0.5,
                        "point_3": 0.0,
                        "point_4": 0.5,
                        "point_5": 1.0,
                    }
                ]
            ).to_csv(path, index=False)
            rows = fe.load_fe_reference(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].basis, "strain")
        self.assertAlmostEqual(np.max(np.abs(rows[0].shape)), 1.0, places=6)
        self.assertGreater(rows[0].shape[0], 0.0)

    def test_ssi_cov_recovers_primary_mode_from_synthetic_system(self) -> None:
        rng = np.random.default_rng(123)
        fs = 50.0
        n_samples = 4000
        mode_shape_1 = np.array([1.0, 0.8, 0.6, 0.35, 0.2], dtype=float)
        mode_shape_2 = np.array([0.1, 0.25, 0.5, 0.8, 1.0], dtype=float)
        comp_1 = _simulate_modal_component(n_samples=n_samples, fs=fs, freq_hz=2.35, damping_ratio=0.02, rng=rng)
        comp_2 = _simulate_modal_component(n_samples=n_samples, fs=fs, freq_hz=3.8, damping_ratio=0.04, rng=rng)
        outputs = (
            np.outer(comp_1, mode_shape_1)
            + 0.6 * np.outer(comp_2, mode_shape_2)
            + 0.02 * rng.normal(size=(n_samples, 5))
        )

        poles = ssi.run_ssi_cov(
            outputs,
            sampling_rate=fs,
            block_rows=20,
            min_order=2,
            max_order=12,
            freq_min=1.0,
            freq_max=5.0,
            max_damping_ratio=0.1,
        )
        poles = ssi.label_stable_poles(poles, freq_tol_hz=0.15, damping_tol=0.05, mac_tol=0.75)
        poles = ssi.assign_mode_clusters(poles, freq_tol_hz=0.15, mac_tol=0.75)
        selected = ssi.select_dominant_cluster(poles, reference_frequency_hz=2.35, focus_min=2.0, focus_max=2.8)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(selected["frequency_hz"], 2.35, delta=0.2)
        self.assertAlmostEqual(selected["damping_ratio"], 0.02, delta=0.03)
        self.assertGreater(spectral.compute_mac(selected["mode_shape"], spectral.normalize_mode_shape(mode_shape_1)), 0.7)

    def test_harmonic_mask_prevents_selecting_strong_harmonic_peak(self) -> None:
        fs = 50.0
        t = np.arange(0.0, 40.0, 1.0 / fs)
        shape = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=float)
        structural = np.sin(2 * np.pi * 2.35 * t)
        harmonic = 2.4 * np.sin(2 * np.pi * 3.6 * t)
        matrix = np.column_stack(
            [shape[i] * structural + harmonic + 0.05 * np.sin(2 * np.pi * (i + 1) * 0.1 * t) for i in range(5)]
        )
        freqs, csd_matrix, _ = spectral.compute_spectral_matrices(matrix, sampling_rate=fs, nperseg=len(t))
        singular_values, _ = spectral.compute_fdd_spectrum(csd_matrix)

        raw_peak_index = spectral.select_peak_index(
            freqs,
            singular_values[:, 0],
            freq_min=1.0,
            freq_max=5.0,
            focus_min=1.0,
            focus_max=5.0,
        )
        keep_mask, _ = spectral.build_harmonic_mask(freqs, rpm=72.0, harmonic_orders=(1, 2, 3, 4), half_width=0.15)
        masked_peak_index = spectral.select_peak_index(
            freqs,
            singular_values[:, 0],
            freq_min=1.0,
            freq_max=5.0,
            focus_min=1.0,
            focus_max=5.0,
            keep_mask=keep_mask,
        )

        self.assertIsNotNone(raw_peak_index)
        self.assertIsNotNone(masked_peak_index)
        assert raw_peak_index is not None and masked_peak_index is not None
        self.assertAlmostEqual(freqs[raw_peak_index], 3.6, delta=0.2)
        self.assertAlmostEqual(freqs[masked_peak_index], 2.35, delta=0.25)

    def test_repo_smoke_run_generates_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            outputs = pipeline.run_modal_identification(
                case_ids=[1, 10, 17],
                output_dir=output_dir,
                sensor_basis="both",
                rpm_source="manifest",
                config=models.ModalConfig(
                    window_seconds=20.0,
                    step_seconds=10.0,
                    ssi_block_rows=20,
                    ssi_max_order=12,
                ),
            )

            required_files = [
                "case_modal_summary.csv",
                "window_modal_estimates.csv",
                "harmonic_mask_table.csv",
                "strain_mode_shapes.csv",
                "accy_mode_shapes.csv",
                "stabilization_poles.csv",
                "stability_statistics.csv",
            ]
            for file_name in required_files:
                self.assertTrue((output_dir / file_name).exists(), file_name)

            summary = outputs["case_modal_summary"]
            self.assertEqual(summary["case_id"].tolist(), [1, 10, 17])
            self.assertTrue(summary["strain_valid_window_count"].gt(0).all())
            self.assertIn("accy_first_frequency_hz", summary.columns)

    def test_save_mode_shape_animation_writes_gif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "case_01_strain_mode_shape_animation.gif"
            saved_path = animation.save_mode_shape_animation(
                case_id=1,
                basis_name="strain",
                mode_shape=np.array([1.0, 0.7, 0.2, -0.3, -0.8], dtype=float),
                output_path=output_path,
                fps=8,
                cycles=2,
            )
            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_mode_shape_animations_skips_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = animation.save_mode_shape_animations(
                output_dir=Path(temp_dir),
                shape_tables={
                    "strain": pd.DataFrame(
                        [
                            {
                                "case_id": 1,
                                "basis": "strain",
                                "frequency_hz": 2.3,
                                "damping_ratio": 0.01,
                                "valid_window_count": 0,
                                "point_1": 1.0,
                                "point_2": 0.7,
                                "point_3": 0.2,
                                "point_4": -0.3,
                                "point_5": -0.8,
                            }
                        ]
                    )
                },
                animation_format="gif",
                fps=8,
                cycles=2,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "skipped")
            self.assertEqual(rows[0]["message"], "no valid windows")

    def test_cli_without_animation_flag_does_not_create_animation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            argv = [
                "python",
                "--case-ids",
                "1",
                "--sensor-basis",
                "strain",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch("sys.argv", argv):
                modal_cli.main()

            self.assertTrue((output_dir / "case_modal_summary.csv").exists())
            self.assertFalse(any(output_dir.glob("*_mode_shape_animation.*")))

    def test_cli_with_animation_flag_creates_animation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            argv = [
                "python",
                "--case-ids",
                "1",
                "--sensor-basis",
                "strain",
                "--output-dir",
                str(output_dir),
                "--save-mode-shape-animation",
                "--animation-format",
                "gif",
                "--animation-fps",
                "8",
                "--animation-cycles",
                "2",
            ]
            with mock.patch("sys.argv", argv):
                modal_cli.main()

            generated = list(output_dir.glob("*_mode_shape_animation.gif"))
            self.assertEqual(len(generated), 1)
            self.assertGreater(generated[0].stat().st_size, 0)


def _simulate_modal_component(
    *,
    n_samples: int,
    fs: float,
    freq_hz: float,
    damping_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    omega_n = 2.0 * np.pi * freq_hz
    omega_d = omega_n * np.sqrt(max(1e-6, 1.0 - damping_ratio**2))
    radius = np.exp(-damping_ratio * omega_n / fs)
    theta = omega_d / fs
    denom = [1.0, -2.0 * radius * np.cos(theta), radius**2]
    excitation = rng.normal(size=n_samples)
    return signal.lfilter([1.0], denom, excitation)


if __name__ == "__main__":
    unittest.main()
