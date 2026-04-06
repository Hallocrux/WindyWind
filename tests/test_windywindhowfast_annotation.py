from __future__ import annotations

import unittest

from src.windyWindHowfast.analysis_core import select_preferred_peak
from src.windyWindHowfast.app import build_arg_parser
from src.windyWindHowfast.main import main


class WindyWindHowfastVideoTests(unittest.TestCase):
    def test_parser_accepts_manual_roi_args(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--video",
                "data/video/demo.mp4",
                "--center-x",
                "10",
                "--center-y",
                "20",
                "--radius",
                "30",
                "--no-show",
            ]
        )
        self.assertEqual(args.center_x, 10)
        self.assertEqual(args.center_y, 20)
        self.assertEqual(args.radius, 30)
        self.assertTrue(args.no_show)

    def test_main_symbol_is_importable(self) -> None:
        self.assertTrue(callable(main))

    def test_peak_selection_prefers_abs_k_3(self) -> None:
        peaks = [
            {"rank": 1, "temporal_hz": 7.0, "spatial_mode_k": -2, "magnitude": 100.0, "rpm": 210.0},
            {"rank": 2, "temporal_hz": 7.0, "spatial_mode_k": -3, "magnitude": 80.0, "rpm": 140.0},
        ]
        selected = select_preferred_peak(peaks, preferred_abs_k=(3,))
        self.assertIsNotNone(selected)
        self.assertEqual(selected["spatial_mode_k"], -3)


if __name__ == "__main__":
    unittest.main()
