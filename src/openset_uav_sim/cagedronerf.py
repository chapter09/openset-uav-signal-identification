from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .signals import normalize_power
from .types import Location, Segment, SegmentMetadata, SemanticOutcome


CAGEDRONERF_SAMPLING_RATE_HZ = 20_000_000.0
CAGEDRONERF_CENTER_FREQUENCY_HZ = 2_447_000_000.0
CAGEDRONERF_BANDWIDTH_HZ = 20_000_000.0


@dataclass(frozen=True)
class CageDroneRFConfig:
    """Loader settings for the CageDroneRF/U-RAPTOR `.dat` layout."""

    raw_root: str | Path
    metadata_path: str | Path | None = None
    segment_length: int = 4096
    stride: int = 4096
    sample_rate_hz: float = CAGEDRONERF_SAMPLING_RATE_HZ
    default_center_frequency_hz: float = CAGEDRONERF_CENTER_FREQUENCY_HZ
    default_bandwidth_hz: float = CAGEDRONERF_BANDWIDTH_HZ
    receiver_id: str = "cagedronerf-rx"
    gain_db: float = 50.0
    antenna: str = "cage-rx"
    receiver_location: Location = field(default_factory=lambda: Location(0.0, 0.0, 0.0))
    default_snr_db: float = 10.0
    remove_dc: bool = True
    normalize: bool = True
    mmap: bool = True


@dataclass(frozen=True)
class CageDroneRFRecording:
    """Metadata parsed from a CageDroneRF raw recording filename."""

    dat_file_path: Path
    label: str
    manufacturer: str
    model: str
    bandwidth_hz: float
    center_frequency_hz: float
    operation_mode: str
    category: str = "drone"


def load_dat_complex64(file_path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load a CageDroneRF raw `.dat` file as complex64 I/Q samples."""

    path = Path(file_path)
    if mmap:
        return np.memmap(path, dtype=np.complex64, mode="r")
    return np.fromfile(path, dtype=np.complex64)


def parse_cagedronerf_recording(file_path: str | Path, raw_root: str | Path | None = None) -> CageDroneRFRecording:
    """Parse CageDroneRF metadata from a raw `.dat` filename.

    The public toolkit expects names like
    `Manufacture_Model_Bandwidth_CenterFreq_Mode.dat`, with special folders such as
    `non-drone/` and `multi-drone/`.
    """

    path = Path(file_path)
    category = _infer_category(path, Path(raw_root) if raw_root is not None else None)
    tokens = path.stem.split("_")
    first_numeric = _first_numeric_index(tokens)

    if category in {"non-drone", "multi-drone"}:
        label_tokens = tokens[:first_numeric] if first_numeric > 0 else tokens[:1]
        label = _clean_label("_".join(label_tokens) or path.stem)
        manufacturer = ""
        model = label
        bandwidth_token = _join_token_slice(tokens, first_numeric, first_numeric + (2 if category == "multi-drone" else 1))
        center_token = _join_token_slice(tokens, first_numeric + (2 if category == "multi-drone" else 1), first_numeric + (4 if category == "multi-drone" else 2))
        mode = _join_token_slice(tokens, first_numeric + (4 if category == "multi-drone" else 2), len(tokens))
    else:
        manufacturer = tokens[0] if tokens else ""
        model = tokens[1] if len(tokens) > 1 else manufacturer or path.stem
        label_tokens = tokens[:first_numeric] if first_numeric > 0 else [manufacturer, model]
        label = _clean_label("_".join(token for token in label_tokens if token) or model)
        bandwidth_token = tokens[first_numeric] if first_numeric >= 0 and first_numeric < len(tokens) else ""
        center_token = tokens[first_numeric + 1] if first_numeric >= 0 and first_numeric + 1 < len(tokens) else ""
        mode = "_".join(tokens[first_numeric + 2 :]) if first_numeric >= 0 and first_numeric + 2 < len(tokens) else ""

    if label == "Skydio":
        label = "Skydio_2"
        model = "Skydio_2"

    return CageDroneRFRecording(
        dat_file_path=path,
        label=label,
        manufacturer=manufacturer,
        model=model,
        bandwidth_hz=_coerce_bandwidth_hz(bandwidth_token, CAGEDRONERF_BANDWIDTH_HZ),
        center_frequency_hz=_coerce_center_frequency_hz(center_token, CAGEDRONERF_CENTER_FREQUENCY_HZ),
        operation_mode=mode,
        category=category,
    )


class CageDroneRFLoader:
    """Bridge CageDroneRF raw recordings into simulator `Segment` objects."""

    def __init__(
        self,
        config: CageDroneRFConfig,
        known_labels: Iterable[str] | None = None,
        unknown_labels: Iterable[str] | None = None,
    ) -> None:
        self.config = config
        self.raw_root = Path(config.raw_root)
        self.metadata_path = Path(config.metadata_path) if config.metadata_path is not None else None
        self.known_labels = {_clean_label(label) for label in known_labels or ()}
        self.unknown_labels = {_clean_label(label) for label in unknown_labels or ()}
        self._dat_index: dict[str, Path] | None = None

    def iter_recordings(self, patterns: Iterable[str] = ("*.dat", "*/*.dat", "**/*.dat")) -> list[CageDroneRFRecording]:
        seen: set[Path] = set()
        recordings: list[CageDroneRFRecording] = []
        for pattern in patterns:
            for path in sorted(self.raw_root.glob(pattern)):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                recordings.append(parse_cagedronerf_recording(path, self.raw_root))
        return recordings

    def load_segments(
        self,
        max_segments_per_recording: int | None = None,
        max_segments_per_label: int | None = None,
    ) -> list[Segment]:
        if self.metadata_path is not None and self.metadata_path.exists():
            segments = self._load_segments_from_metadata(max_segments_per_recording, max_segments_per_label)
        else:
            segments = self._load_segments_from_raw(max_segments_per_recording, max_segments_per_label)
        return segments

    def make_open_set_splits(
        self,
        train_fraction: float = 0.6,
        val_fraction: float = 0.2,
        seed: int = 2026,
        max_segments_per_recording: int | None = None,
        max_segments_per_label: int | None = None,
    ) -> dict[str, list[Segment]]:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1).")
        if train_fraction + val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be less than 1.")

        rng = np.random.default_rng(seed)
        by_label: dict[str, list[Segment]] = {}
        for segment in self.load_segments(max_segments_per_recording, max_segments_per_label):
            by_label.setdefault(segment.label, []).append(segment)

        train: list[Segment] = []
        val: list[Segment] = []
        test: list[Segment] = []
        for label, label_segments in sorted(by_label.items()):
            rng.shuffle(label_segments)
            if any(segment.is_unknown for segment in label_segments):
                test.extend(label_segments)
                continue
            train_end = max(1, int(round(len(label_segments) * train_fraction)))
            val_end = train_end + int(round(len(label_segments) * val_fraction))
            train.extend(label_segments[:train_end])
            val.extend(label_segments[train_end:val_end])
            test.extend(label_segments[val_end:])

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        return {"train": train, "val": val, "test": test}

    def _load_segments_from_raw(
        self,
        max_segments_per_recording: int | None,
        max_segments_per_label: int | None,
    ) -> list[Segment]:
        segments: list[Segment] = []
        label_counts: dict[str, int] = {}
        for recording in self.iter_recordings():
            if _limit_reached(label_counts, recording.label, max_segments_per_label):
                continue
            signal = load_dat_complex64(recording.dat_file_path, mmap=self.config.mmap)
            per_recording_count = 0
            for start in range(0, max(0, signal.size - self.config.segment_length + 1), self.config.stride):
                if max_segments_per_recording is not None and per_recording_count >= max_segments_per_recording:
                    break
                if _limit_reached(label_counts, recording.label, max_segments_per_label):
                    break
                iq = np.array(signal[start : start + self.config.segment_length], dtype=np.complex64)
                segments.append(self._make_segment(iq, recording, start, start + self.config.segment_length))
                per_recording_count += 1
                label_counts[recording.label] = label_counts.get(recording.label, 0) + 1
        return segments

    def _load_segments_from_metadata(
        self,
        max_segments_per_recording: int | None,
        max_segments_per_label: int | None,
    ) -> list[Segment]:
        records = _load_metadata_records(self.metadata_path)
        segments: list[Segment] = []
        label_counts: dict[str, int] = {}
        recording_counts: dict[str, int] = {}
        for record in records:
            dat_path = self._resolve_dat_path(str(record.get("dat_file_path", "")))
            if dat_path is None:
                continue
            recording = parse_cagedronerf_recording(dat_path, self.raw_root)
            recording_key = str(dat_path)
            if max_segments_per_recording is not None and recording_counts.get(recording_key, 0) >= max_segments_per_recording:
                continue
            if _limit_reached(label_counts, recording.label, max_segments_per_label):
                continue
            start = int(record.get("start_index", 0))
            end = int(record.get("end_index", start + self.config.segment_length))
            if end <= start:
                end = start + self.config.segment_length
            signal = load_dat_complex64(dat_path, mmap=self.config.mmap)
            if start >= signal.size:
                continue
            iq = np.array(signal[start : min(end, signal.size)], dtype=np.complex64)
            iq = _pad_or_trim(iq, self.config.segment_length)
            segment = self._make_segment(iq, recording, start, start + len(iq), metadata_record=record)
            segments.append(segment)
            recording_counts[recording_key] = recording_counts.get(recording_key, 0) + 1
            label_counts[recording.label] = label_counts.get(recording.label, 0) + 1
        return segments

    def _make_segment(
        self,
        iq: np.ndarray,
        recording: CageDroneRFRecording,
        start_index: int,
        end_index: int,
        metadata_record: dict[str, Any] | None = None,
    ) -> Segment:
        iq = _preprocess_iq(iq, remove_dc=self.config.remove_dc, normalize=self.config.normalize)
        outcome = self._outcome_for_recording(recording)
        cluster_id = f"cagedronerf-unknown-{recording.label}" if outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER else None
        estimated_snr_db = float((metadata_record or {}).get("snr_db", self.config.default_snr_db))
        timestamp_s = float(start_index / self.config.sample_rate_hz)
        metadata = SegmentMetadata(
            center_frequency_hz=recording.center_frequency_hz or self.config.default_center_frequency_hz,
            bandwidth_hz=recording.bandwidth_hz or self.config.default_bandwidth_hz,
            timestamp_s=timestamp_s,
            receiver_id=self.config.receiver_id,
            gain_db=self.config.gain_db,
            location=self.config.receiver_location,
            antenna=self.config.antenna,
            estimated_snr_db=estimated_snr_db,
        )
        return Segment(
            iq=iq,
            metadata=metadata,
            outcome=outcome,
            label=recording.label,
            emitter_id=None if outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER else recording.label,
            cluster_id=cluster_id,
            sequence_id=str(recording.dat_file_path),
            burst_index=start_index // max(self.config.segment_length, 1),
        )

    def _outcome_for_recording(self, recording: CageDroneRFRecording) -> SemanticOutcome:
        label = _clean_label(recording.label)
        if recording.category == "non-drone":
            if _is_background_label(label, recording.operation_mode):
                return SemanticOutcome.TRUE_BACKGROUND_NOISE
            return SemanticOutcome.KNOWN_NON_UAV_EMITTER
        if label in self.unknown_labels:
            return SemanticOutcome.UNKNOWN_UAV_CLUSTER
        if self.known_labels and label not in self.known_labels:
            return SemanticOutcome.UNKNOWN_UAV_CLUSTER
        return SemanticOutcome.KNOWN_UAV_ID

    def _resolve_dat_path(self, dat_file_path: str) -> Path | None:
        if not dat_file_path:
            return None
        direct = Path(dat_file_path)
        if direct.exists():
            return direct
        relative = self.raw_root / dat_file_path
        if relative.exists():
            return relative
        index = self._get_dat_index()
        return index.get(Path(dat_file_path).name)

    def _get_dat_index(self) -> dict[str, Path]:
        if self._dat_index is None:
            self._dat_index = {path.name: path for path in self.raw_root.glob("**/*.dat")}
        return self._dat_index


def _load_metadata_records(metadata_path: Path | None) -> list[dict[str, Any]]:
    if metadata_path is None:
        return []
    with metadata_path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
        return [record for record in records if isinstance(record, dict)]
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def _preprocess_iq(iq: np.ndarray, remove_dc: bool, normalize: bool) -> np.ndarray:
    x = np.asarray(iq, dtype=np.complex64)
    if remove_dc and x.size:
        x = x - np.mean(x)
    if normalize and x.size:
        x = normalize_power(x)
    return x.astype(np.complex64)


def _pad_or_trim(iq: np.ndarray, segment_length: int) -> np.ndarray:
    if iq.size == segment_length:
        return iq.astype(np.complex64)
    if iq.size > segment_length:
        return iq[:segment_length].astype(np.complex64)
    padded = np.zeros(segment_length, dtype=np.complex64)
    padded[: iq.size] = iq
    return padded


def _limit_reached(label_counts: dict[str, int], label: str, max_segments_per_label: int | None) -> bool:
    return max_segments_per_label is not None and label_counts.get(label, 0) >= max_segments_per_label


def _infer_category(path: Path, raw_root: Path | None) -> str:
    parts = path.parts
    if raw_root is not None:
        try:
            parts = path.relative_to(raw_root).parts
        except ValueError:
            parts = path.parts
    folder = parts[0].lower() if len(parts) > 1 else ""
    if folder in {"non-drone", "non_drone", "nodrone", "no-drone"}:
        return "non-drone"
    if folder in {"multi-drone", "multi_drone", "multidrone"}:
        return "multi-drone"
    return "drone"


def _first_numeric_index(tokens: list[str]) -> int:
    for index, token in enumerate(tokens):
        if _to_float(token) is not None:
            return index
    return -1


def _join_token_slice(tokens: list[str], start: int, end: int) -> str:
    if start < 0:
        return ""
    return "_".join(tokens[start:end])


def _clean_label(label: str) -> str:
    return label.strip().replace(" ", "_")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    for separator in ("-", "_"):
        if separator in text:
            for part in text.split(separator):
                parsed = _to_float(part)
                if parsed is not None:
                    return parsed
            return None
    text = text.replace("MHz", "").replace("mhz", "").replace("GHz", "").replace("ghz", "")
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_center_frequency_hz(value: Any, default_hz: float) -> float:
    parsed = _to_float(value)
    if parsed is None or parsed <= 0:
        return default_hz
    if parsed < 10:
        return parsed * 1e9
    if parsed < 10_000:
        return parsed * 1e6
    if parsed < 10_000_000:
        return parsed * 1e3
    return parsed


def _coerce_bandwidth_hz(value: Any, default_hz: float) -> float:
    parsed = _to_float(value)
    if parsed is None or parsed <= 0:
        return default_hz
    if parsed < 1_000:
        return parsed * 1e6
    if parsed < 10_000_000:
        return parsed * 1e3
    return parsed


def _is_background_label(label: str, operation_mode: str) -> bool:
    text = f"{label}_{operation_mode}".lower().replace("-", "_")
    return any(token in text for token in ("no_drone", "nodrone", "background", "noise"))
