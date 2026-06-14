from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .prototype import Prediction
from .types import Segment, SemanticOutcome


@dataclass(frozen=True)
class CurvePoint:
    x: float
    y: float
    threshold: float


@dataclass(frozen=True)
class OpenSetMetrics:
    known_accuracy: float
    unknown_rejection_rate: float
    overall_accuracy: float
    macro_f1_known: float
    auroc: float
    oscr: float
    known_count: int
    unknown_count: int
    total_count: int


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    count: int
    correct: int
    rejected_as_unknown: int
    accuracy: float


@dataclass(frozen=True)
class OpenSetReport:
    metrics: OpenSetMetrics
    per_label: list[LabelMetrics]
    roc_curve: list[CurvePoint]
    oscr_curve: list[CurvePoint]
    confusion_labels: list[str]
    confusion_matrix: list[list[int]]
    energy_known: list[float]
    energy_unknown: list[float]


def evaluate_predictions(segments: Sequence[Segment], predictions: Sequence[Prediction]) -> OpenSetReport:
    if len(segments) != len(predictions):
        raise ValueError("segments and predictions must have the same length.")
    if not segments:
        raise ValueError("At least one segment is required for evaluation.")

    known_pairs = [(segment, prediction) for segment, prediction in zip(segments, predictions) if not segment.is_unknown]
    unknown_pairs = [(segment, prediction) for segment, prediction in zip(segments, predictions) if segment.is_unknown]
    known_count = len(known_pairs)
    unknown_count = len(unknown_pairs)
    total_count = len(segments)

    known_correct = sum(
        1
        for segment, prediction in known_pairs
        if prediction.accepted and prediction.label == segment.training_label
    )
    unknown_rejected = sum(1 for _, prediction in unknown_pairs if not prediction.accepted)
    total_correct = known_correct + unknown_rejected

    known_accuracy = known_correct / known_count if known_count else math.nan
    unknown_rejection_rate = unknown_rejected / unknown_count if unknown_count else math.nan
    overall_accuracy = total_correct / total_count

    true_known_labels = sorted({segment.training_label for segment, _ in known_pairs})
    per_label = _per_label_metrics(true_known_labels, known_pairs)
    macro_f1_known = _macro_f1(true_known_labels, known_pairs)

    energy_known = [float(prediction.energy) for _, prediction in known_pairs]
    energy_unknown = [float(prediction.energy) for _, prediction in unknown_pairs]
    auroc = binary_auroc(energy_known, energy_unknown)
    roc_curve = roc_points(energy_known, energy_unknown)
    oscr_curve = oscr_points(known_pairs, unknown_pairs)
    oscr = area_under_curve([(point.x, point.y) for point in oscr_curve])

    confusion_labels, confusion_matrix = confusion_matrix_for_predictions(segments, predictions)
    return OpenSetReport(
        metrics=OpenSetMetrics(
            known_accuracy=known_accuracy,
            unknown_rejection_rate=unknown_rejection_rate,
            overall_accuracy=overall_accuracy,
            macro_f1_known=macro_f1_known,
            auroc=auroc,
            oscr=oscr,
            known_count=known_count,
            unknown_count=unknown_count,
            total_count=total_count,
        ),
        per_label=per_label,
        roc_curve=roc_curve,
        oscr_curve=oscr_curve,
        confusion_labels=confusion_labels,
        confusion_matrix=confusion_matrix,
        energy_known=energy_known,
        energy_unknown=energy_unknown,
    )


def binary_auroc(energy_known: Sequence[float], energy_unknown: Sequence[float]) -> float:
    """AUROC with unknown as the positive class and energy as the score."""

    negatives = np.asarray(energy_known, dtype=np.float64)
    positives = np.asarray(energy_unknown, dtype=np.float64)
    if negatives.size == 0 or positives.size == 0:
        return math.nan
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros(negatives.size, dtype=np.int8), np.ones(positives.size, dtype=np.int8)])
    ranks = _average_ranks(scores)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    positive_count = positives.size
    negative_count = negatives.size
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def roc_points(energy_known: Sequence[float], energy_unknown: Sequence[float]) -> list[CurvePoint]:
    """ROC points for unknown detection, using higher energy as more unknown-like."""

    known = np.asarray(energy_known, dtype=np.float64)
    unknown = np.asarray(energy_unknown, dtype=np.float64)
    if known.size == 0 or unknown.size == 0:
        return []
    thresholds = _descending_thresholds(np.concatenate([known, unknown]))
    points: list[CurvePoint] = []
    for threshold in thresholds:
        true_positive_rate = float(np.mean(unknown >= threshold))
        false_positive_rate = float(np.mean(known >= threshold))
        points.append(CurvePoint(x=false_positive_rate, y=true_positive_rate, threshold=float(threshold)))
    return sorted(points, key=lambda point: (point.x, point.y))


def oscr_points(
    known_pairs: Sequence[tuple[Segment, Prediction]],
    unknown_pairs: Sequence[tuple[Segment, Prediction]],
) -> list[CurvePoint]:
    """Open-set classification-rate curve.

    Lower GE-OSR energy means "more known-like"; a threshold accepts samples with
    energy <= threshold. X is unknown false-positive rate, Y is known correct
    classification rate.
    """

    if not known_pairs or not unknown_pairs:
        return []
    energies = np.asarray(
        [prediction.energy for _, prediction in known_pairs]
        + [prediction.energy for _, prediction in unknown_pairs],
        dtype=np.float64,
    )
    thresholds = _ascending_thresholds(energies)
    known_total = len(known_pairs)
    unknown_total = len(unknown_pairs)
    points: list[CurvePoint] = []
    for threshold in thresholds:
        correct_known = sum(
            1
            for segment, prediction in known_pairs
            if prediction.energy <= threshold and prediction.label == segment.training_label
        )
        accepted_unknown = sum(1 for _, prediction in unknown_pairs if prediction.energy <= threshold)
        points.append(
            CurvePoint(
                x=accepted_unknown / unknown_total,
                y=correct_known / known_total,
                threshold=float(threshold),
            )
        )
    return _upper_envelope(points)


def area_under_curve(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.nan
    ordered = sorted(points)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered[:-1], ordered[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return float(area)


def confusion_matrix_for_predictions(
    segments: Sequence[Segment],
    predictions: Sequence[Prediction],
) -> tuple[list[str], list[list[int]]]:
    true_labels = [segment.training_label if not segment.is_unknown else "unknown" for segment in segments]
    pred_labels = [prediction.label if prediction.accepted else "unknown" for prediction in predictions]
    labels = sorted(set(true_labels) | set(pred_labels))
    if "unknown" in labels:
        labels = [label for label in labels if label != "unknown"] + ["unknown"]
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[index[true_label]][index[pred_label]] += 1
    return labels, matrix


def save_open_set_report(report: OpenSetReport, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "metrics.json", _report_to_dict(report))
    _write_metrics_csv(output_path / "metrics_summary.csv", report.metrics)
    _write_metrics_markdown(output_path / "metrics_summary.md", report.metrics)
    _write_per_label_csv(output_path / "per_label_metrics.csv", report.per_label)
    _write_per_label_markdown(output_path / "per_label_metrics.md", report.per_label)
    _write_curve_csv(output_path / "roc_curve.csv", report.roc_curve, x_name="fpr", y_name="tpr")
    _write_curve_csv(output_path / "oscr_curve.csv", report.oscr_curve, x_name="unknown_fpr", y_name="known_ccr")
    _write_confusion_csv(output_path / "confusion_matrix.csv", report.confusion_labels, report.confusion_matrix)
    _write_energy_histogram_svg(output_path / "energy_histogram.svg", report.energy_known, report.energy_unknown)
    _write_curve_svg(output_path / "roc_curve.svg", report.roc_curve, "ROC Curve", "False Positive Rate", "True Positive Rate")
    _write_curve_svg(output_path / "oscr_curve.svg", report.oscr_curve, "OSCR Curve", "Unknown False Positive Rate", "Known CCR")
    _write_confusion_svg(output_path / "confusion_matrix.svg", report.confusion_labels, report.confusion_matrix)


def _per_label_metrics(
    labels: Sequence[str],
    known_pairs: Sequence[tuple[Segment, Prediction]],
) -> list[LabelMetrics]:
    metrics: list[LabelMetrics] = []
    for label in labels:
        pairs = [(segment, prediction) for segment, prediction in known_pairs if segment.training_label == label]
        count = len(pairs)
        correct = sum(1 for segment, prediction in pairs if prediction.accepted and prediction.label == segment.training_label)
        rejected = sum(1 for _, prediction in pairs if not prediction.accepted)
        metrics.append(
            LabelMetrics(
                label=label,
                count=count,
                correct=correct,
                rejected_as_unknown=rejected,
                accuracy=correct / count if count else math.nan,
            )
        )
    return metrics


def _macro_f1(labels: Sequence[str], known_pairs: Sequence[tuple[Segment, Prediction]]) -> float:
    if not labels:
        return math.nan
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(
            1
            for segment, prediction in known_pairs
            if segment.training_label == label and prediction.accepted and prediction.label == label
        )
        fp = sum(
            1
            for segment, prediction in known_pairs
            if segment.training_label != label and prediction.accepted and prediction.label == label
        )
        fn = sum(
            1
            for segment, prediction in known_pairs
            if segment.training_label == label and (not prediction.accepted or prediction.label != label)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1_scores))


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _ascending_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    return np.concatenate([[np.nextafter(unique[0], -np.inf)], unique, [np.nextafter(unique[-1], np.inf)]])


def _descending_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)[::-1]
    return np.concatenate([[np.nextafter(unique[0], np.inf)], unique, [np.nextafter(unique[-1], -np.inf)]])


def _upper_envelope(points: Sequence[CurvePoint]) -> list[CurvePoint]:
    by_x: dict[float, CurvePoint] = {}
    for point in points:
        current = by_x.get(point.x)
        if current is None or point.y > current.y:
            by_x[point.x] = point
    return [by_x[x] for x in sorted(by_x)]


def _report_to_dict(report: OpenSetReport) -> dict[str, object]:
    return {
        "metrics": asdict(report.metrics),
        "per_label": [asdict(item) for item in report.per_label],
        "roc_curve": [asdict(item) for item in report.roc_curve],
        "oscr_curve": [asdict(item) for item in report.oscr_curve],
        "confusion_labels": report.confusion_labels,
        "confusion_matrix": report.confusion_matrix,
        "energy_known": report.energy_known,
        "energy_unknown": report.energy_unknown,
    }


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_metrics_csv(path: Path, metrics: OpenSetMetrics) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in asdict(metrics).items():
            writer.writerow([key, _format_number(value)])


def _write_metrics_markdown(path: Path, metrics: OpenSetMetrics) -> None:
    rows = ["| Metric | Value |", "|---|---:|"]
    for key, value in asdict(metrics).items():
        rows.append(f"| {key} | {_format_number(value)} |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_per_label_csv(path: Path, per_label: Sequence[LabelMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "count", "correct", "rejected_as_unknown", "accuracy"])
        writer.writeheader()
        for item in per_label:
            writer.writerow(asdict(item))


def _write_per_label_markdown(path: Path, per_label: Sequence[LabelMetrics]) -> None:
    rows = ["| Label | Count | Correct | Rejected as Unknown | Accuracy |", "|---|---:|---:|---:|---:|"]
    for item in per_label:
        rows.append(
            f"| {item.label} | {item.count} | {item.correct} | {item.rejected_as_unknown} | {_format_number(item.accuracy)} |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_curve_csv(path: Path, points: Sequence[CurvePoint], x_name: str, y_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_name, y_name, "threshold"])
        for point in points:
            writer.writerow([point.x, point.y, point.threshold])


def _write_confusion_csv(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_energy_histogram_svg(path: Path, known: Sequence[float], unknown: Sequence[float]) -> None:
    width, height = 760, 460
    margin = 62
    known_arr = np.asarray(known, dtype=np.float64)
    unknown_arr = np.asarray(unknown, dtype=np.float64)
    all_values = np.concatenate([known_arr, unknown_arr]) if known_arr.size or unknown_arr.size else np.array([0.0, 1.0])
    low, high = float(np.min(all_values)), float(np.max(all_values))
    if low == high:
        low -= 0.5
        high += 0.5
    bins = np.linspace(low, high, 25)
    known_counts, _ = np.histogram(known_arr, bins=bins)
    unknown_counts, _ = np.histogram(unknown_arr, bins=bins)
    max_count = max(int(np.max(known_counts, initial=0)), int(np.max(unknown_counts, initial=0)), 1)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bar_w = plot_w / (len(bins) - 1)
    elements = [_svg_header(width, height), _svg_axes(margin, width, height, "Energy", "Count")]
    for idx, count in enumerate(known_counts):
        x = margin + idx * bar_w
        h = plot_h * count / max_count
        y = height - margin - h
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.46:.2f}" height="{h:.2f}" fill="#2563eb" opacity="0.68"/>')
    for idx, count in enumerate(unknown_counts):
        x = margin + idx * bar_w + bar_w * 0.48
        h = plot_h * count / max_count
        y = height - margin - h
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.46:.2f}" height="{h:.2f}" fill="#dc2626" opacity="0.68"/>')
    elements.extend(
        [
            '<text x="62" y="34" font-size="20" font-family="Arial" font-weight="700">GE-OSR Energy Distribution</text>',
            '<rect x="540" y="28" width="16" height="16" fill="#2563eb" opacity="0.68"/><text x="564" y="41" font-size="13" font-family="Arial">Known</text>',
            '<rect x="620" y="28" width="16" height="16" fill="#dc2626" opacity="0.68"/><text x="644" y="41" font-size="13" font-family="Arial">Unknown</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_curve_svg(path: Path, points: Sequence[CurvePoint], title: str, x_label: str, y_label: str) -> None:
    width, height = 620, 460
    margin = 62
    elements = [_svg_header(width, height), _svg_axes(margin, width, height, x_label, y_label)]
    if points:
        coords = []
        for point in points:
            x = margin + point.x * (width - 2 * margin)
            y = height - margin - point.y * (height - 2 * margin)
            coords.append(f"{x:.2f},{y:.2f}")
        elements.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="#0f766e" stroke-width="3"/>')
        for coord in coords[:: max(1, len(coords) // 24)]:
            x, y = coord.split(",")
            elements.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#0f766e"/>')
    elements.append(f'<text x="{margin}" y="34" font-size="20" font-family="Arial" font-weight="700">{_escape_xml(title)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_confusion_svg(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    cell = 42
    label_w = 160
    top = 90
    width = label_w + cell * len(labels) + 40
    height = top + cell * len(labels) + 80
    max_count = max([value for row in matrix for value in row], default=1)
    elements = [_svg_header(width, height)]
    elements.append('<text x="24" y="34" font-size="20" font-family="Arial" font-weight="700">Confusion Matrix</text>')
    for col, label in enumerate(labels):
        x = label_w + col * cell + cell / 2
        elements.append(
            f'<text x="{x:.2f}" y="76" font-size="10" font-family="Arial" text-anchor="middle" transform="rotate(-35 {x:.2f} 76)">{_escape_xml(label[:18])}</text>'
        )
    for row_idx, label in enumerate(labels):
        y = top + row_idx * cell + cell / 2 + 4
        elements.append(f'<text x="{label_w - 8}" y="{y:.2f}" font-size="11" font-family="Arial" text-anchor="end">{_escape_xml(label[:24])}</text>')
        for col_idx, value in enumerate(matrix[row_idx]):
            intensity = value / max_count if max_count else 0.0
            color = _blue_scale(intensity)
            x = label_w + col_idx * cell
            yy = top + row_idx * cell
            elements.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            text_color = "#ffffff" if intensity > 0.52 else "#0f172a"
            elements.append(
                f'<text x="{x + cell / 2:.2f}" y="{yy + cell / 2 + 4:.2f}" font-size="11" font-family="Arial" text-anchor="middle" fill="{text_color}">{value}</text>'
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _svg_axes(margin: int, width: int, height: int, x_label: str, y_label: str) -> str:
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    return "\n".join(
        [
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#1f2937" stroke-width="1.4"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#1f2937" stroke-width="1.4"/>',
            f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 18}" font-size="13" font-family="Arial" text-anchor="middle">{_escape_xml(x_label)}</text>',
            f'<text x="18" y="{(y0 + y1) / 2:.2f}" font-size="13" font-family="Arial" text-anchor="middle" transform="rotate(-90 18 {(y0 + y1) / 2:.2f})">{_escape_xml(y_label)}</text>',
            f'<text x="{x0}" y="{y0 + 18}" font-size="10" font-family="Arial" text-anchor="middle">0</text>',
            f'<text x="{x1}" y="{y0 + 18}" font-size="10" font-family="Arial" text-anchor="middle">1</text>',
            f'<text x="{x0 - 14}" y="{y0 + 4}" font-size="10" font-family="Arial" text-anchor="end">0</text>',
            f'<text x="{x0 - 14}" y="{y1 + 4}" font-size="10" font-family="Arial" text-anchor="end">1</text>',
        ]
    )


def _blue_scale(intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    light = np.array([239, 246, 255])
    dark = np.array([37, 99, 235])
    rgb = (light * (1.0 - intensity) + dark * intensity).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_number(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)

