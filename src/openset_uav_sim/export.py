from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import Segment


def segments_to_iq_array(segments: list[Segment]) -> np.ndarray:
    """Return I/Q as an array with shape [segments, samples, 2]."""

    if not segments:
        return np.empty((0, 0, 2), dtype=np.float32)
    iq = np.stack([segment.iq for segment in segments])
    return np.stack([np.real(iq), np.imag(iq)], axis=-1).astype(np.float32)


def save_segments(segments: list[Segment], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records = [segment.to_record() for segment in segments]
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        iq=segments_to_iq_array(segments),
        labels=np.array([segment.label for segment in segments], dtype=object),
        outcomes=np.array([segment.outcome.value for segment in segments], dtype=object),
        emitter_ids=np.array([segment.emitter_id or "" for segment in segments], dtype=object),
        cluster_ids=np.array([segment.cluster_id or "" for segment in segments], dtype=object),
    )
    with output_prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def save_splits(splits: dict[str, list[Segment]], output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"splits": {}}
    for split_name, segments in splits.items():
        save_segments(segments, output_path / split_name)
        split_summary: dict[str, object] = {
            "count": len(segments),
            "outcomes": {},
            "labels": {},
        }
        for segment in segments:
            outcomes = split_summary["outcomes"]
            labels = split_summary["labels"]
            assert isinstance(outcomes, dict)
            assert isinstance(labels, dict)
            outcomes[segment.outcome.value] = outcomes.get(segment.outcome.value, 0) + 1
            labels[segment.label] = labels.get(segment.label, 0) + 1
        summary["splits"][split_name] = split_summary

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary

