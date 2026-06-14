from __future__ import annotations

import unittest

from openset_uav_sim.geosr import GEOSRConfig, TORCH_AVAILABLE, TorchUnavailableError, require_torch


class GEOSRConfigTests(unittest.TestCase):
    def test_paper_hyperparameter_defaults(self) -> None:
        config = GEOSRConfig()

        self.assertEqual(config.alpha, 32.0)
        self.assertEqual(config.delta, 0.1)
        self.assertEqual(config.temperature, 10.0)
        self.assertEqual(config.target_energy, -0.1)
        self.assertEqual(config.lambda_dce, 0.3)
        self.assertEqual(config.lambda_fea, 1.0)
        self.assertEqual(config.beta, 0.2)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.learning_rate, 0.001)

    def test_missing_torch_reports_actionable_message(self) -> None:
        if TORCH_AVAILABLE:
            self.skipTest("PyTorch is installed in this environment.")

        with self.assertRaises(TorchUnavailableError) as context:
            require_torch()
        self.assertIn("GE-OSR requires PyTorch", str(context.exception))


if __name__ == "__main__":
    unittest.main()
