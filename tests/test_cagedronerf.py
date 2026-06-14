from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openset_uav_sim import CageDroneRFConfig, CageDroneRFLoader, SemanticOutcome
from openset_uav_sim.cagedronerf import parse_cagedronerf_recording


class CageDroneRFTests(unittest.TestCase):
    def test_parse_drone_filename(self) -> None:
        recording = parse_cagedronerf_recording("DJI_Mavic3_20_2447_hover.dat")

        self.assertEqual(recording.label, "DJI_Mavic3")
        self.assertEqual(recording.manufacturer, "DJI")
        self.assertEqual(recording.model, "Mavic3")
        self.assertEqual(recording.bandwidth_hz, 20_000_000.0)
        self.assertEqual(recording.center_frequency_hz, 2_447_000_000.0)
        self.assertEqual(recording.operation_mode, "hover")

    def test_load_raw_dat_as_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = (np.arange(64, dtype=np.float32) + 1j * np.arange(64, dtype=np.float32)).astype(np.complex64)
            signal.tofile(root / "DJI_Mavic3_20_2447_hover.dat")

            loader = CageDroneRFLoader(
                CageDroneRFConfig(raw_root=root, segment_length=16, stride=16),
                unknown_labels=["DJI_Mavic3"],
            )
            segments = loader.load_segments(max_segments_per_recording=2)

            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0].iq.shape, (16,))
            self.assertEqual(segments[0].outcome, SemanticOutcome.UNKNOWN_UAV_CLUSTER)
            self.assertEqual(segments[0].cluster_id, "cagedronerf-unknown-DJI_Mavic3")
            self.assertEqual(segments[0].metadata.center_frequency_hz, 2_447_000_000.0)

    def test_metadata_file_can_reference_raw_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = (np.ones(32, dtype=np.float32) + 1j * np.zeros(32, dtype=np.float32)).astype(np.complex64)
            dat_path = root / "non-drone" / "NoDrone_20_2447_idle.dat"
            dat_path.parent.mkdir()
            signal.tofile(dat_path)
            metadata_path = root / "meta_data.json"
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "dat_file_path": str(dat_path),
                            "start_index": 4,
                            "end_index": 20,
                            "snr_db": -15.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loader = CageDroneRFLoader(CageDroneRFConfig(raw_root=root, metadata_path=metadata_path, segment_length=16))
            segments = loader.load_segments()

            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].outcome, SemanticOutcome.TRUE_BACKGROUND_NOISE)
            self.assertEqual(segments[0].metadata.estimated_snr_db, -15.0)

    def test_known_labels_do_not_turn_non_drone_into_unknown_uav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            non_drone = root / "non-drone"
            non_drone.mkdir()
            signal = (np.ones(32, dtype=np.float32) + 1j * np.zeros(32, dtype=np.float32)).astype(np.complex64)
            signal.tofile(non_drone / "WiFi_20_2447_idle.dat")

            loader = CageDroneRFLoader(
                CageDroneRFConfig(raw_root=root, segment_length=16),
                known_labels=["DJI_Mavic3"],
            )
            segments = loader.load_segments(max_segments_per_recording=1)

            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].outcome, SemanticOutcome.KNOWN_NON_UAV_EMITTER)


if __name__ == "__main__":
    unittest.main()
