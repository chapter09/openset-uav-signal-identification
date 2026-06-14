from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openset_uav_sim.evaluation import binary_auroc, evaluate_predictions, save_open_set_report
from openset_uav_sim.prototype import Prediction
from openset_uav_sim.types import Location, Segment, SegmentMetadata, SemanticOutcome


class EvaluationTests(unittest.TestCase):
    def test_binary_auroc_uses_higher_energy_for_unknown(self) -> None:
        self.assertEqual(binary_auroc([0.1, 0.2, 0.3], [0.7, 0.8]), 1.0)

    def test_evaluate_predictions_and_write_report(self) -> None:
        segments = [
            _segment("known-a", SemanticOutcome.KNOWN_UAV_ID),
            _segment("known-b", SemanticOutcome.KNOWN_UAV_ID),
            _segment("unknown-a", SemanticOutcome.UNKNOWN_UAV_CLUSTER, cluster_id="unknown-a"),
            _segment("unknown-a", SemanticOutcome.UNKNOWN_UAV_CLUSTER, cluster_id="unknown-a"),
        ]
        predictions = [
            _prediction("known-a", SemanticOutcome.KNOWN_UAV_ID, accepted=True, energy=0.1),
            _prediction("unknown", SemanticOutcome.UNKNOWN_UAV_CLUSTER, accepted=False, energy=0.5),
            _prediction("unknown", SemanticOutcome.UNKNOWN_UAV_CLUSTER, accepted=False, energy=0.9),
            _prediction("known-a", SemanticOutcome.KNOWN_UAV_ID, accepted=True, energy=0.2),
        ]

        report = evaluate_predictions(segments, predictions)

        self.assertEqual(report.metrics.known_count, 2)
        self.assertEqual(report.metrics.unknown_count, 2)
        self.assertEqual(report.metrics.known_accuracy, 0.5)
        self.assertEqual(report.metrics.unknown_rejection_rate, 0.5)
        self.assertGreaterEqual(report.metrics.auroc, 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            save_open_set_report(report, tmp)
            expected = [
                "metrics.json",
                "metrics_summary.csv",
                "metrics_summary.md",
                "per_label_metrics.csv",
                "per_label_metrics.md",
                "energy_histogram.svg",
                "roc_curve.svg",
                "oscr_curve.svg",
                "confusion_matrix.svg",
            ]
            for name in expected:
                self.assertTrue((Path(tmp) / name).exists(), name)


def _segment(label: str, outcome: SemanticOutcome, cluster_id: str | None = None) -> Segment:
    metadata = SegmentMetadata(
        center_frequency_hz=2.4e9,
        bandwidth_hz=20e6,
        timestamp_s=0.0,
        receiver_id="rx",
        gain_db=0.0,
        location=Location(0.0, 0.0, 0.0),
        antenna="omni",
        estimated_snr_db=10.0,
    )
    return Segment(
        iq=np.zeros(16, dtype=np.complex64),
        metadata=metadata,
        outcome=outcome,
        label=label,
        emitter_id=None if outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER else label,
        cluster_id=cluster_id,
    )


def _prediction(label: str, outcome: SemanticOutcome, accepted: bool, energy: float) -> Prediction:
    return Prediction(
        outcome=outcome,
        label=label,
        energy=energy,
        distance=max(0.0, energy),
        nearest_label=label,
        accepted=accepted,
    )


if __name__ == "__main__":
    unittest.main()
