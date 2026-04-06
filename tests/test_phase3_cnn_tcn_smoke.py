from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "try"
    / "013_phase3_cnn_tcn_smoke"
    / "phase3_cnn_tcn_lib.py"
)
SPEC = importlib.util.spec_from_file_location("phase3_cnn_tcn_lib", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "try" / "009_phase1_feature_groups"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "try" / "012_phase3_end_to_end_shortlist"))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Phase3CnnTcnSmokeTests(unittest.TestCase):
    def test_tiny_cnn_forward_shape(self) -> None:
        model = MODULE.Tiny1DCNN(in_channels=20)
        x = torch.randn(4, 20, 250)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4,))

    def test_tiny_tcn_forward_shape(self) -> None:
        model = MODULE.TinyTCN(in_channels=20)
        x = torch.randn(4, 20, 250)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4,))

    def test_normalize_windows_by_channel(self) -> None:
        X_train = np.random.randn(8, 3, 5).astype(np.float32)
        X_valid = np.random.randn(4, 3, 5).astype(np.float32)
        train_norm, valid_norm = MODULE.normalize_windows_by_channel(X_train, X_valid)
        self.assertEqual(train_norm.shape, X_train.shape)
        self.assertEqual(valid_norm.shape, X_valid.shape)


if __name__ == "__main__":
    unittest.main()
