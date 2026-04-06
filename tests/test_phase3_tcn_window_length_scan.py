from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "try"
    / "014_phase3_tcn_window_length_scan"
    / "run_tcn_window_scan.py"
)
SPEC = importlib.util.spec_from_file_location("run_tcn_window_scan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
repo_root = Path(__file__).resolve().parents[1]
for extra in (
    repo_root,
    repo_root / "src" / "try" / "009_phase1_feature_groups",
    repo_root / "src" / "try" / "012_phase3_end_to_end_shortlist",
    repo_root / "src" / "try" / "013_phase3_cnn_tcn_smoke",
):
    sys.path.insert(0, str(extra))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Phase3TcnWindowScanTests(unittest.TestCase):
    def test_window_configs_are_sorted_and_half_overlap(self) -> None:
        labels = [label for label, _ in MODULE.WINDOW_CONFIGS]
        self.assertEqual(labels, ["2s", "4s", "5s", "8s"])
        for _, config in MODULE.WINDOW_CONFIGS:
            self.assertEqual(config.step_size * 2, config.window_size)


if __name__ == "__main__":
    unittest.main()
