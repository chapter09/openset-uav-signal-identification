from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

import numpy as np

from .signals import (
    complex_noise,
    db_to_linear,
    normalize_power,
    synthesize_clean_waveform,
    apply_emitter_impairments,
)
from .types import (
    EmitterProfile,
    Location,
    Receiver,
    Segment,
    SegmentMetadata,
    SemanticOutcome,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation settings."""

    segment_length: int = 4096
    sample_rate_hz: float = 30_000_000.0
    timestamp_start_s: float = 0.0
    timestamp_step_s: float = 0.02
    snr_jitter_db: float = 4.0
    snr_clip_db: tuple[float, float] = (-12.0, 30.0)
    estimated_snr_error_db: float = 1.25
    receiver_gain_jitter_db: float = 1.5
    unknown_probability: float = 0.25
    background_probability: float = 0.18


class OpenSetUAVEnvironment:
    """Simulation scene for known and unknown UAV RF clusters."""

    def __init__(
        self,
        receivers: Iterable[Receiver],
        emitters: Iterable[EmitterProfile],
        config: SimulationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.receivers = tuple(receivers)
        self.emitters = tuple(emitters)
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self._clock = self.config.timestamp_start_s

        if not self.receivers:
            raise ValueError("At least one receiver is required.")
        if not self.emitters:
            raise ValueError("At least one emitter profile is required.")

    @classmethod
    def default(cls, seed: int | None = 2026) -> "OpenSetUAVEnvironment":
        """Create a repeatable starter scene with known and unknown RF sources."""

        config = SimulationConfig()
        receivers = (
            Receiver(
                receiver_id="rx-rooftop-01",
                location=Location(40.7128, -74.0060, 35.0),
                gain_db=28.0,
                antenna="log-periodic",
                sample_rate_hz=config.sample_rate_hz,
            ),
            Receiver(
                receiver_id="rx-mobile-02",
                location=Location(40.7139, -74.0018, 4.0),
                gain_db=21.0,
                antenna="patch-array",
                sample_rate_hz=config.sample_rate_hz,
            ),
        )
        emitters = (
            EmitterProfile(
                emitter_id="uav-dji-p4rtk-01",
                display_name="Known UAV DJI P4RTK 01",
                outcome=SemanticOutcome.KNOWN_UAV_ID,
                waveform="uav_video",
                center_frequency_hz=2_437_000_000.0,
                bandwidth_hz=8_000_000.0,
                nominal_snr_db=13.0,
                location=Location(40.7144, -74.0048, 110.0),
                symbol_rate_hz=2_000_000.0,
                frequency_offset_hz=1150.0,
                phase_bias_rad=0.34,
                iq_gain_imbalance=0.035,
                iq_phase_imbalance_rad=0.018,
                phase_noise_std=0.004,
                mobility_hz=5.0,
            ),
            EmitterProfile(
                emitter_id="uav-autel-evo-02",
                display_name="Known UAV Autel EVO 02",
                outcome=SemanticOutcome.KNOWN_UAV_ID,
                waveform="controller_qpsk",
                center_frequency_hz=2_462_000_000.0,
                bandwidth_hz=4_000_000.0,
                nominal_snr_db=10.5,
                location=Location(40.7105, -74.0081, 85.0),
                symbol_rate_hz=850_000.0,
                frequency_offset_hz=-720.0,
                phase_bias_rad=-0.28,
                iq_gain_imbalance=-0.025,
                iq_phase_imbalance_rad=-0.022,
                phase_noise_std=0.005,
                burst_duty_cycle=0.72,
                mobility_hz=3.0,
            ),
            EmitterProfile(
                emitter_id="uav-parrot-anafi-03",
                display_name="Known UAV Parrot Anafi 03",
                outcome=SemanticOutcome.KNOWN_UAV_ID,
                waveform="telemetry_fsk",
                center_frequency_hz=915_000_000.0,
                bandwidth_hz=1_200_000.0,
                nominal_snr_db=8.0,
                location=Location(40.7160, -74.0025, 65.0),
                symbol_rate_hz=250_000.0,
                frequency_offset_hz=410.0,
                phase_bias_rad=0.12,
                iq_gain_imbalance=0.015,
                iq_phase_imbalance_rad=0.026,
                phase_noise_std=0.006,
                burst_duty_cycle=0.55,
                mobility_hz=2.0,
            ),
            EmitterProfile(
                emitter_id="wifi-ap-warehouse",
                display_name="Known non-UAV Wi-Fi AP",
                outcome=SemanticOutcome.KNOWN_NON_UAV_EMITTER,
                waveform="wifi_like",
                center_frequency_hz=2_437_000_000.0,
                bandwidth_hz=20_000_000.0,
                nominal_snr_db=16.0,
                location=Location(40.7131, -74.0056, 15.0),
                frequency_offset_hz=95.0,
                phase_bias_rad=-0.04,
                iq_gain_imbalance=0.008,
                iq_phase_imbalance_rad=-0.006,
                phase_noise_std=0.002,
                burst_duty_cycle=0.92,
            ),
            EmitterProfile(
                emitter_id="industrial-ism-sweep",
                display_name="Known non-UAV ISM sweep",
                outcome=SemanticOutcome.KNOWN_NON_UAV_EMITTER,
                waveform="chirp",
                center_frequency_hz=915_000_000.0,
                bandwidth_hz=2_000_000.0,
                nominal_snr_db=11.0,
                location=Location(40.7113, -74.0032, 12.0),
                frequency_offset_hz=-180.0,
                phase_bias_rad=0.51,
                iq_gain_imbalance=-0.018,
                iq_phase_imbalance_rad=0.012,
                phase_noise_std=0.003,
                burst_duty_cycle=0.48,
            ),
            EmitterProfile(
                emitter_id="unknown-uav-alpha-source",
                display_name="Unknown UAV cluster alpha",
                outcome=SemanticOutcome.UNKNOWN_UAV_CLUSTER,
                waveform="uav_video",
                center_frequency_hz=2_452_000_000.0,
                bandwidth_hz=10_000_000.0,
                nominal_snr_db=9.0,
                location=Location(40.7152, -74.0092, 130.0),
                cluster_id="unknown-uav-cluster-alpha",
                symbol_rate_hz=2_600_000.0,
                frequency_offset_hz=1880.0,
                phase_bias_rad=0.76,
                iq_gain_imbalance=0.055,
                iq_phase_imbalance_rad=0.041,
                phase_noise_std=0.007,
                burst_duty_cycle=0.78,
                mobility_hz=6.0,
            ),
            EmitterProfile(
                emitter_id="unknown-uav-beta-controller",
                display_name="Unknown UAV cluster beta",
                outcome=SemanticOutcome.UNKNOWN_UAV_CLUSTER,
                waveform="controller_qpsk",
                center_frequency_hz=5_805_000_000.0,
                bandwidth_hz=5_000_000.0,
                nominal_snr_db=6.5,
                location=Location(40.7089, -74.0069, 95.0),
                cluster_id="unknown-uav-cluster-beta",
                symbol_rate_hz=1_100_000.0,
                frequency_offset_hz=-1460.0,
                phase_bias_rad=-0.63,
                iq_gain_imbalance=-0.045,
                iq_phase_imbalance_rad=-0.034,
                phase_noise_std=0.008,
                burst_duty_cycle=0.66,
                mobility_hz=4.0,
            ),
        )
        return cls(receivers=receivers, emitters=emitters, config=config, seed=seed)

    @property
    def known_emitters(self) -> tuple[EmitterProfile, ...]:
        return tuple(profile for profile in self.emitters if profile.is_known)

    @property
    def unknown_emitters(self) -> tuple[EmitterProfile, ...]:
        return tuple(
            profile
            for profile in self.emitters
            if profile.outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER
        )

    def sample_segment(
        self,
        outcome: SemanticOutcome | None = None,
        emitter_id: str | None = None,
        receiver_id: str | None = None,
        timestamp_s: float | None = None,
        sequence_id: str | None = None,
        burst_index: int | None = None,
    ) -> Segment:
        receiver = self._choose_receiver(receiver_id)
        timestamp = self._next_timestamp() if timestamp_s is None else timestamp_s

        if outcome is None and emitter_id is None and self.rng.random() < self.config.background_probability:
            outcome = SemanticOutcome.TRUE_BACKGROUND_NOISE

        if outcome == SemanticOutcome.TRUE_BACKGROUND_NOISE:
            profile = None
        else:
            profile = self._choose_profile(outcome=outcome, emitter_id=emitter_id)

        if profile is None:
            iq, estimated_snr_db, metadata = self._sample_background(receiver, timestamp)
            return Segment(
                iq=iq,
                metadata=metadata,
                outcome=SemanticOutcome.TRUE_BACKGROUND_NOISE,
                label="background/noise",
                sequence_id=sequence_id,
                burst_index=burst_index,
            )

        iq, estimated_snr_db = self._sample_iq_for_profile(profile, receiver)
        metadata = SegmentMetadata(
            center_frequency_hz=float(profile.center_frequency_hz + self.rng.normal(0.0, profile.bandwidth_hz * 0.002)),
            bandwidth_hz=float(profile.bandwidth_hz * self.rng.uniform(0.96, 1.04)),
            timestamp_s=float(timestamp),
            receiver_id=receiver.receiver_id,
            gain_db=float(receiver.gain_db + self.rng.normal(0.0, self.config.receiver_gain_jitter_db)),
            location=receiver.location,
            antenna=receiver.antenna,
            estimated_snr_db=float(estimated_snr_db),
        )
        label = profile.cluster_id if profile.outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER else profile.emitter_id
        return Segment(
            iq=iq,
            metadata=metadata,
            outcome=profile.outcome,
            label=label or profile.emitter_id,
            emitter_id=profile.emitter_id,
            cluster_id=profile.cluster_id,
            sequence_id=sequence_id,
            burst_index=burst_index,
        )

    def sample_burst_sequence(
        self,
        burst_count: int,
        outcome: SemanticOutcome | None = None,
        emitter_id: str | None = None,
        receiver_id: str | None = None,
    ) -> list[Segment]:
        if burst_count <= 0:
            raise ValueError("burst_count must be positive.")
        profile = None if outcome == SemanticOutcome.TRUE_BACKGROUND_NOISE else self._choose_profile(outcome, emitter_id)
        sequence_id = f"seq-{uuid4().hex[:12]}"
        timestamp = self._next_timestamp()
        segments: list[Segment] = []
        for index in range(burst_count):
            segments.append(
                self.sample_segment(
                    outcome=SemanticOutcome.TRUE_BACKGROUND_NOISE if profile is None else profile.outcome,
                    emitter_id=None if profile is None else profile.emitter_id,
                    receiver_id=receiver_id,
                    timestamp_s=timestamp + index * self.config.timestamp_step_s,
                    sequence_id=sequence_id,
                    burst_index=index,
                )
            )
        self._clock = max(self._clock, timestamp + burst_count * self.config.timestamp_step_s)
        return segments

    def sample_dataset(
        self,
        count: int,
        include_unknown: bool = True,
        include_background: bool = True,
        balanced: bool = True,
    ) -> list[Segment]:
        if count < 0:
            raise ValueError("count must be non-negative.")
        choices: list[EmitterProfile | SemanticOutcome] = list(self.known_emitters)
        if include_unknown:
            choices.extend(self.unknown_emitters)
        if include_background:
            choices.append(SemanticOutcome.TRUE_BACKGROUND_NOISE)
        if not choices:
            raise ValueError("No choices available for dataset sampling.")

        segments: list[Segment] = []
        for index in range(count):
            if balanced:
                choice = choices[index % len(choices)]
            else:
                choice = self._weighted_choice(include_unknown, include_background)
            if isinstance(choice, SemanticOutcome):
                segments.append(self.sample_segment(outcome=choice))
            else:
                segments.append(self.sample_segment(outcome=choice.outcome, emitter_id=choice.emitter_id))
        self.rng.shuffle(segments)
        return segments

    def make_open_set_splits(
        self,
        train_per_known: int = 48,
        val_per_known: int = 16,
        test_per_known: int = 24,
        unknown_per_cluster: int = 24,
        background_per_split: int = 48,
    ) -> dict[str, list[Segment]]:
        """Create train/val/test splits for open-set experiments."""

        train: list[Segment] = []
        val: list[Segment] = []
        test: list[Segment] = []

        for profile in self.known_emitters:
            train.extend(
                self.sample_segment(outcome=profile.outcome, emitter_id=profile.emitter_id)
                for _ in range(train_per_known)
            )
            val.extend(
                self.sample_segment(outcome=profile.outcome, emitter_id=profile.emitter_id)
                for _ in range(val_per_known)
            )
            test.extend(
                self.sample_segment(outcome=profile.outcome, emitter_id=profile.emitter_id)
                for _ in range(test_per_known)
            )

        train.extend(self.sample_segment(outcome=SemanticOutcome.TRUE_BACKGROUND_NOISE) for _ in range(background_per_split))
        val.extend(self.sample_segment(outcome=SemanticOutcome.TRUE_BACKGROUND_NOISE) for _ in range(max(1, background_per_split // 3)))
        test.extend(self.sample_segment(outcome=SemanticOutcome.TRUE_BACKGROUND_NOISE) for _ in range(background_per_split))

        for profile in self.unknown_emitters:
            test.extend(
                self.sample_segment(outcome=profile.outcome, emitter_id=profile.emitter_id)
                for _ in range(unknown_per_cluster)
            )

        self.rng.shuffle(train)
        self.rng.shuffle(val)
        self.rng.shuffle(test)
        return {"train": train, "val": val, "test": test}

    def _sample_iq_for_profile(self, profile: EmitterProfile, receiver: Receiver) -> tuple[np.ndarray, float]:
        n = self.config.segment_length
        clean = synthesize_clean_waveform(self.rng, profile, n, receiver.sample_rate_hz)
        impaired = apply_emitter_impairments(self.rng, clean, profile, receiver.sample_rate_hz)
        target_snr_db = float(
            np.clip(
                profile.nominal_snr_db + self.rng.normal(0.0, self.config.snr_jitter_db),
                self.config.snr_clip_db[0],
                self.config.snr_clip_db[1],
            )
        )
        signal = normalize_power(impaired)
        noise_power = 1.0 / db_to_linear(target_snr_db)
        noise = complex_noise(self.rng, n, noise_power)
        gain = 10 ** (receiver.gain_db / 20.0)
        adc_scale = 1.0 / max(gain, 1.0)
        iq = (signal + noise) * adc_scale
        iq = self._apply_receiver_artifacts(iq)
        estimated_snr_db = target_snr_db + float(self.rng.normal(0.0, self.config.estimated_snr_error_db))
        return iq.astype(np.complex64), estimated_snr_db

    def _sample_background(self, receiver: Receiver, timestamp_s: float) -> tuple[np.ndarray, float, SegmentMetadata]:
        n = self.config.segment_length
        noise_power = float(self.rng.uniform(0.6, 1.4))
        iq = complex_noise(self.rng, n, noise_power)
        iq = self._apply_receiver_artifacts(iq)
        estimated_snr_db = float(self.rng.normal(-18.0, 2.5))
        center_frequency_hz = float(self.rng.choice([915_000_000.0, 2_437_000_000.0, 5_805_000_000.0]))
        metadata = SegmentMetadata(
            center_frequency_hz=center_frequency_hz,
            bandwidth_hz=float(receiver.sample_rate_hz * self.rng.uniform(0.15, 0.8)),
            timestamp_s=float(timestamp_s),
            receiver_id=receiver.receiver_id,
            gain_db=float(receiver.gain_db + self.rng.normal(0.0, self.config.receiver_gain_jitter_db)),
            location=receiver.location,
            antenna=receiver.antenna,
            estimated_snr_db=estimated_snr_db,
        )
        return iq.astype(np.complex64), estimated_snr_db, metadata

    def _apply_receiver_artifacts(self, iq: np.ndarray) -> np.ndarray:
        dc = self.rng.normal(0.0, 0.015) + 1j * self.rng.normal(0.0, 0.015)
        clipped = np.clip(np.real(iq + dc), -1.5, 1.5) + 1j * np.clip(np.imag(iq + dc), -1.5, 1.5)
        return clipped.astype(np.complex64)

    def _choose_receiver(self, receiver_id: str | None) -> Receiver:
        if receiver_id is None:
            return self.receivers[int(self.rng.integers(0, len(self.receivers)))]
        for receiver in self.receivers:
            if receiver.receiver_id == receiver_id:
                return receiver
        raise KeyError(f"Unknown receiver_id: {receiver_id}")

    def _choose_profile(
        self,
        outcome: SemanticOutcome | None = None,
        emitter_id: str | None = None,
    ) -> EmitterProfile | None:
        if emitter_id is not None:
            for profile in self.emitters:
                if profile.emitter_id == emitter_id:
                    return profile
            raise KeyError(f"Unknown emitter_id: {emitter_id}")

        if outcome == SemanticOutcome.TRUE_BACKGROUND_NOISE:
            return None

        candidates = self.emitters
        if outcome is not None:
            candidates = tuple(profile for profile in self.emitters if profile.outcome == outcome)
        else:
            candidates = tuple(
                profile
                for profile in self.emitters
                if profile.outcome != SemanticOutcome.UNKNOWN_UAV_CLUSTER
            )
            if self.rng.random() < self.config.unknown_probability:
                candidates = self.unknown_emitters

        if not candidates:
            raise ValueError(f"No emitter profiles available for outcome={outcome!r}.")
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _weighted_choice(
        self,
        include_unknown: bool,
        include_background: bool,
    ) -> EmitterProfile | SemanticOutcome:
        if include_background and self.rng.random() < self.config.background_probability:
            return SemanticOutcome.TRUE_BACKGROUND_NOISE
        if include_unknown and self.rng.random() < self.config.unknown_probability:
            return self.unknown_emitters[int(self.rng.integers(0, len(self.unknown_emitters)))]
        return self.known_emitters[int(self.rng.integers(0, len(self.known_emitters)))]

    def _next_timestamp(self) -> float:
        timestamp = self._clock
        self._clock += self.config.timestamp_step_s
        return float(timestamp)
