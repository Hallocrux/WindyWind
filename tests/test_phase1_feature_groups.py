from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "try"
    / "009_phase1_feature_groups"
    / "phase1_feature_groups_lib.py"
)
SPEC = importlib.util.spec_from_file_location("phase1_feature_groups_lib", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PHASE1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PHASE1
SPEC.loader.exec_module(PHASE1)


class Phase1FeatureGroupTests(unittest.TestCase):
    def test_channel_superset_uses_raw_mean_for_detrending(self) -> None:
        raw_signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        feature_map, _ = PHASE1.extract_channel_feature_superset(
            raw_signal,
            prefix="sig_a",
            sampling_rate=50.0,
        )
        self.assertAlmostEqual(feature_map["sig_a__raw_mean"], 2.5, places=6)
        self.assertAlmostEqual(feature_map["sig_a__raw_median"], 2.5, places=6)
        self.assertAlmostEqual(feature_map["sig_a__dyn_min"], -1.5, places=6)
        self.assertAlmostEqual(feature_map["sig_a__dyn_max"], 1.5, places=6)
        self.assertAlmostEqual(feature_map["sig_a__dyn_ptp"], 3.0, places=6)

    def test_zero_signal_frequency_features_fall_back_to_zero(self) -> None:
        raw_signal = np.zeros(16, dtype=float)
        feature_map, _ = PHASE1.extract_channel_feature_superset(
            raw_signal,
            prefix="sig_zero",
            sampling_rate=50.0,
        )
        self.assertEqual(feature_map["sig_zero__dyn_fft_peak_freq"], 0.0)
        self.assertEqual(feature_map["sig_zero__dyn_fft_peak_amp"], 0.0)
        self.assertEqual(feature_map["sig_zero__dyn_fft_total_energy"], 0.0)
        self.assertEqual(feature_map["sig_zero__dyn_spectral_centroid"], 0.0)
        self.assertEqual(feature_map["sig_zero__dyn_fft_top1_freq"], 0.0)

    def test_single_frequency_top1_matches_signal(self) -> None:
        sampling_rate = 50.0
        time_axis = np.arange(0, 1.0, 1.0 / sampling_rate)
        raw_signal = 5.0 + np.sin(2 * np.pi * 5.0 * time_axis)
        feature_map, _ = PHASE1.extract_channel_feature_superset(
            raw_signal,
            prefix="sig_sine",
            sampling_rate=sampling_rate,
        )
        self.assertAlmostEqual(feature_map["sig_sine__raw_mean"], 5.0, places=5)
        self.assertAlmostEqual(feature_map["sig_sine__dyn_fft_peak_freq"], 5.0, places=5)
        self.assertAlmostEqual(feature_map["sig_sine__dyn_fft_top1_freq"], 5.0, places=5)

    def test_cross_channel_summary_uses_dynamic_signals(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
        _, cache_x = PHASE1.extract_channel_feature_superset(x, "WSMS00001.AccX", 50.0)
        _, cache_y = PHASE1.extract_channel_feature_superset(y, "WSMS00002.AccX", 50.0)
        features = PHASE1.build_cross_channel_features(
            {"WSMS00001.AccX": cache_x, "WSMS00002.AccX": cache_y}
        )
        self.assertAlmostEqual(features["acc_corr_mean"], 1.0, places=6)
        self.assertAlmostEqual(features["strain_corr_mean"], 0.0, places=6)

    def test_group_feature_columns_include_only_raw_mean_and_raw_median_in_baseline_block(self) -> None:
        feature_df = pd.DataFrame(
            [
                {
                    "sig_a__raw_mean": 1.0,
                    "sig_a__raw_median": 1.0,
                    "sig_a__dyn_std": 0.5,
                    "sig_a__dyn_min": -1.0,
                    "sig_a__dyn_max": 1.0,
                    "sig_a__dyn_ptp": 2.0,
                    "sig_a__dyn_rms": 0.7,
                    "sig_a__dyn_fft_peak_freq": 5.0,
                    "sig_a__dyn_fft_peak_amp": 3.0,
                    "sig_a__dyn_fft_total_energy": 9.0,
                    "sig_a__dyn_fft_band_ratio_0_2hz": 0.1,
                    "sig_a__dyn_fft_band_ratio_2_5hz": 0.2,
                    "sig_a__dyn_fft_band_ratio_5_10hz": 0.7,
                    "raw_missing_ratio": 0.0,
                    "raw_missing_rows": 0,
                    "touches_leading_missing": 0,
                    "touches_trailing_missing": 0,
                }
            ]
        )
        columns = PHASE1.get_group_feature_columns(feature_df, "G0_BASE")
        self.assertIn("sig_a__raw_mean", columns)
        self.assertIn("sig_a__raw_median", columns)
        self.assertNotIn("sig_a__mean", columns)
        self.assertNotIn("sig_a__median", columns)

    def test_feature_manifest_matches_group_selector(self) -> None:
        feature_df = pd.DataFrame(
            [
                {
                    "sig_a__raw_mean": 1.0,
                    "sig_a__raw_median": 1.0,
                    "sig_a__dyn_std": 0.5,
                    "sig_a__dyn_min": -1.0,
                    "sig_a__dyn_max": 1.0,
                    "sig_a__dyn_ptp": 2.0,
                    "sig_a__dyn_rms": 0.7,
                    "sig_a__dyn_fft_peak_freq": 5.0,
                    "sig_a__dyn_fft_peak_amp": 3.0,
                    "sig_a__dyn_fft_total_energy": 9.0,
                    "sig_a__dyn_fft_band_ratio_0_2hz": 0.1,
                    "sig_a__dyn_fft_band_ratio_2_5hz": 0.2,
                    "sig_a__dyn_fft_band_ratio_5_10hz": 0.7,
                    "raw_missing_ratio": 0.0,
                    "raw_missing_rows": 0,
                    "touches_leading_missing": 0,
                    "touches_trailing_missing": 0,
                }
            ]
        )
        manifest = PHASE1.build_feature_manifest(feature_df)
        g0_columns = manifest.loc[manifest["group_name"] == "G0_BASE", "feature_column"].tolist()
        self.assertEqual(g0_columns, PHASE1.get_group_feature_columns(feature_df, "G0_BASE"))

    def test_select_finalists_obeys_thresholds(self) -> None:
        screening_summary = pd.DataFrame(
            [
                {"group_name": "G0_BASE", "feature_count": 10, "case_mae": 0.50, "case_rmse": 0.80, "case_mape": 10.0},
                {"group_name": "G1_ROBUST_TIME", "feature_count": 12, "case_mae": 0.48, "case_rmse": 0.79, "case_mape": 9.0},
                {"group_name": "G2_FREQ_SHAPE", "feature_count": 13, "case_mae": 0.49, "case_rmse": 0.81, "case_mape": 9.5},
            ]
        )
        finalists = PHASE1.select_finalists(screening_summary)
        self.assertEqual(finalists["group_name"].tolist(), ["G1_ROBUST_TIME"])

    def test_evaluate_promotion_detects_failure(self) -> None:
        screening_summary = pd.DataFrame(
            [
                {"group_name": "G0_BASE", "feature_count": 10, "case_mae": 0.50, "case_rmse": 0.80, "case_mape": 10.0},
                {"group_name": "G1_ROBUST_TIME", "feature_count": 12, "case_mae": 0.48, "case_rmse": 0.79, "case_mape": 9.0},
            ]
        )
        screening_cases = pd.DataFrame(
            [
                {"feature_set": "G1_ROBUST_TIME", "case_id": 1, "abs_error_delta_vs_base": 0.0},
                {"feature_set": "G1_ROBUST_TIME", "case_id": 2, "abs_error_delta_vs_base": 0.11},
                {"feature_set": "G1_ROBUST_TIME", "case_id": 3, "abs_error_delta_vs_base": 0.12},
                {"feature_set": "G1_ROBUST_TIME", "case_id": 4, "abs_error_delta_vs_base": 0.13},
                {"feature_set": "G1_ROBUST_TIME", "case_id": 5, "abs_error_delta_vs_base": 0.14},
            ]
        )
        finalists = pd.DataFrame(
            [{"group_name": "G1_ROBUST_TIME", "feature_count": 12, "case_mae": 0.48, "case_rmse": 0.79}]
        )
        finalist_summary = pd.DataFrame(
            [
                {"group_name": "G1_ROBUST_TIME", "task_mode": "rpm_free", "model_name": "Ridge", "case_mae": 0.50, "case_rmse": 0.80},
                {"group_name": "G1_ROBUST_TIME", "task_mode": "rpm_free", "model_name": "RandomForestRegressor", "case_mae": 0.48, "case_rmse": 0.79},
            ]
        )
        decision = PHASE1.evaluate_promotion(
            screening_summary=screening_summary,
            screening_cases=screening_cases,
            finalists=finalists,
            finalist_summary=finalist_summary,
            unlabeled_prediction=3.2,
        )
        self.assertFalse(decision.promoted)
        self.assertTrue(decision.fail_reasons)


if __name__ == "__main__":
    unittest.main()
