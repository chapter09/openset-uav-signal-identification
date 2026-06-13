from __future__ import annotations

import unittest

from openset_uav_sim import OpenSetUAVEnvironment, PrototypeOpenSetModel, SemanticOutcome


class EnvironmentTests(unittest.TestCase):
    def test_segment_has_expected_shape_and_metadata(self) -> None:
        env = OpenSetUAVEnvironment.default(seed=11)
        segment = env.sample_segment(outcome=SemanticOutcome.KNOWN_UAV_ID)

        self.assertEqual(segment.iq.shape, (env.config.segment_length,))
        self.assertTrue(segment.iq.dtype.kind == "c")
        metadata = segment.metadata.to_dict()
        self.assertIn("center_frequency_hz", metadata)
        self.assertIn("bandwidth_hz", metadata)
        self.assertIn("timestamp_s", metadata)
        self.assertIn("receiver_id", metadata)
        self.assertIn("gain_db", metadata)
        self.assertIn("location", metadata)
        self.assertIn("antenna", metadata)
        self.assertIn("estimated_snr_db", metadata)

    def test_open_set_splits_hold_out_unknown_clusters_from_training(self) -> None:
        env = OpenSetUAVEnvironment.default(seed=12)
        splits = env.make_open_set_splits(
            train_per_known=3,
            val_per_known=2,
            test_per_known=2,
            unknown_per_cluster=2,
            background_per_split=3,
        )

        self.assertFalse(any(segment.is_unknown for segment in splits["train"]))
        self.assertTrue(any(segment.is_unknown for segment in splits["test"]))
        self.assertTrue(any(segment.outcome == SemanticOutcome.TRUE_BACKGROUND_NOISE for segment in splits["train"]))

    def test_reference_prototype_model_predicts_semantic_outcome(self) -> None:
        env = OpenSetUAVEnvironment.default(seed=13)
        splits = env.make_open_set_splits(
            train_per_known=5,
            val_per_known=2,
            test_per_known=2,
            unknown_per_cluster=2,
            background_per_split=5,
        )
        model = PrototypeOpenSetModel(tail_quantile=0.9)
        model.fit(splits["train"])
        prediction = model.predict(splits["test"][0])

        self.assertIn(
            prediction.outcome,
            {
                SemanticOutcome.KNOWN_UAV_ID,
                SemanticOutcome.KNOWN_NON_UAV_EMITTER,
                SemanticOutcome.TRUE_BACKGROUND_NOISE,
                SemanticOutcome.UNKNOWN_UAV_CLUSTER,
            },
        )
        self.assertGreaterEqual(prediction.distance, 0.0)


if __name__ == "__main__":
    unittest.main()

