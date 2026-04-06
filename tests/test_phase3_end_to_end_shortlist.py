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
    / "012_phase3_end_to_end_shortlist"
    / "phase3_end_to_end_lib.py"
)
SPEC = importlib.util.spec_from_file_location("phase3_end_to_end_lib", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "try" / "009_phase1_feature_groups"))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Phase3EndToEndTests(unittest.TestCase):
    def test_flatten_windows_shape(self) -> None:
        X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        flat = MODULE._flatten_windows(X)
        self.assertEqual(flat.shape, (2, 12))

    def test_minirocket_like_transform_shape(self) -> None:
        model = MODULE.MiniRocketLikeRidge(random_state=42)
        X = np.ones((5, 2, 20), dtype=float)
        features = model.transform(X)
        self.assertEqual(features.shape[0], 5)
        self.assertEqual(features.shape[1], 2 * len(model.kernels) * 2)

    def test_summarize_predictions(self) -> None:
        prediction_frame = pd.DataFrame(
            [
                {"case_id": 1, "file_name": "a.csv", "true_wind_speed": 3.0, "pred_wind_speed": 2.5},
                {"case_id": 1, "file_name": "a.csv", "true_wind_speed": 3.0, "pred_wind_speed": 3.5},
                {"case_id": 2, "file_name": "b.csv", "true_wind_speed": 4.0, "pred_wind_speed": 4.5},
            ]
        )
        summary_row, case_df = MODULE.summarize_predictions(prediction_frame, "RawFlattenRidge")
        self.assertEqual(case_df.shape[0], 2)
        self.assertAlmostEqual(summary_row["case_mae"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
