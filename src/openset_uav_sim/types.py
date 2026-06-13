from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np


class SemanticOutcome(str, Enum):
    """Semantic output classes for open-set UAV signal identification."""

    KNOWN_UAV_ID = "known_uav_id"
    KNOWN_NON_UAV_EMITTER = "known_non_uav_emitter"
    TRUE_BACKGROUND_NOISE = "true_background_noise"
    UNKNOWN_UAV_CLUSTER = "unknown_uav_cluster"


@dataclass(frozen=True)
class Location:
    """Receiver or emitter location."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Receiver:
    """RF receiver description."""

    receiver_id: str
    location: Location
    gain_db: float
    antenna: str
    sample_rate_hz: float = 30_000_000.0
    noise_figure_db: float = 5.0


@dataclass(frozen=True)
class EmitterProfile:
    """A physical or semantic emitter source in the simulation scene."""

    emitter_id: str
    display_name: str
    outcome: SemanticOutcome
    waveform: str
    center_frequency_hz: float
    bandwidth_hz: float
    nominal_snr_db: float
    location: Location
    cluster_id: str | None = None
    antenna: str = "omni"
    symbol_rate_hz: float = 1_000_000.0
    burst_duty_cycle: float = 0.85
    frequency_offset_hz: float = 0.0
    phase_bias_rad: float = 0.0
    iq_gain_imbalance: float = 0.0
    iq_phase_imbalance_rad: float = 0.0
    phase_noise_std: float = 0.003
    mobility_hz: float = 0.0

    @property
    def is_known(self) -> bool:
        return self.outcome in {
            SemanticOutcome.KNOWN_UAV_ID,
            SemanticOutcome.KNOWN_NON_UAV_EMITTER,
        }

    @property
    def is_uav(self) -> bool:
        return self.outcome in {
            SemanticOutcome.KNOWN_UAV_ID,
            SemanticOutcome.UNKNOWN_UAV_CLUSTER,
        }


@dataclass(frozen=True)
class SegmentMetadata:
    """Metadata carried by each I/Q segment."""

    center_frequency_hz: float
    bandwidth_hz: float
    timestamp_s: float
    receiver_id: str
    gain_db: float
    location: Location
    antenna: str
    estimated_snr_db: float

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["location"] = self.location.to_dict()
        return record


@dataclass(frozen=True)
class Segment:
    """A simulated complex I/Q segment with labels and metadata."""

    iq: np.ndarray
    metadata: SegmentMetadata
    outcome: SemanticOutcome
    label: str
    emitter_id: str | None = None
    cluster_id: str | None = None
    sequence_id: str | None = None
    burst_index: int | None = None

    @property
    def is_unknown(self) -> bool:
        return self.outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER

    @property
    def is_background(self) -> bool:
        return self.outcome == SemanticOutcome.TRUE_BACKGROUND_NOISE

    @property
    def training_label(self) -> str:
        """Return the class label used for prototype-style training."""

        if self.outcome == SemanticOutcome.TRUE_BACKGROUND_NOISE:
            return "background/noise"
        return self.label

    def to_record(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "label": self.label,
            "emitter_id": self.emitter_id,
            "cluster_id": self.cluster_id,
            "sequence_id": self.sequence_id,
            "burst_index": self.burst_index,
            "metadata": self.metadata.to_dict(),
        }

