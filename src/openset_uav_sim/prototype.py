from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import extract_features
from .types import Segment, SemanticOutcome


@dataclass(frozen=True)
class Prediction:
    """Semantic prediction from the reference prototype model."""

    outcome: SemanticOutcome
    label: str
    energy: float
    distance: float
    nearest_label: str
    accepted: bool


class PrototypeOpenSetModel:
    """Reference open-set model using prototype distances and energy scores.

    This is intentionally small. It is a sanity-check baseline inspired by prototype and
    energy OSR methods, not a replacement for a trainable deep embedding model.
    """

    def __init__(
        self,
        tail_quantile: float = 0.97,
        threshold_margin: float = 0.08,
        temperature: float = 0.25,
    ) -> None:
        if not 0.5 <= tail_quantile < 1.0:
            raise ValueError("tail_quantile must be in [0.5, 1.0).")
        self.tail_quantile = tail_quantile
        self.threshold_margin = threshold_margin
        self.temperature = temperature
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.prototypes: dict[str, np.ndarray] = {}
        self.class_radii: dict[str, float] = {}
        self.label_outcomes: dict[str, SemanticOutcome] = {}

    def fit(self, segments: list[Segment]) -> "PrototypeOpenSetModel":
        known = [segment for segment in segments if not segment.is_unknown]
        if not known:
            raise ValueError("At least one non-unknown segment is required for fitting.")

        labels = [segment.training_label for segment in known]
        raw_features = np.vstack([extract_features(segment.iq) for segment in known])
        self.feature_mean = np.mean(raw_features, axis=0)
        self.feature_std = np.std(raw_features, axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        features = self._scale(raw_features)

        self.prototypes.clear()
        self.class_radii.clear()
        self.label_outcomes.clear()

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            indices = [index for index, item in enumerate(labels) if item == label]
            class_features = features[indices]
            prototype = np.mean(class_features, axis=0)
            distances = np.linalg.norm(class_features - prototype, axis=1)
            radius = float(np.quantile(distances, self.tail_quantile))
            self.prototypes[label] = prototype
            self.class_radii[label] = max(radius * (1.0 + self.threshold_margin), 1e-6)
            self.label_outcomes[label] = known[indices[0]].outcome

        all_min_distances = [
            self._nearest_distance(feature)[1]
            for feature in features
        ]
        global_radius = float(np.quantile(all_min_distances, self.tail_quantile))
        for label in self.class_radii:
            self.class_radii[label] = max(self.class_radii[label], global_radius * 0.5)
        return self

    def predict(self, segment: Segment) -> Prediction:
        self._require_fit()
        feature = self._scale(extract_features(segment.iq)[None, :])[0]
        nearest_label, distance = self._nearest_distance(feature)
        radius = self.class_radii[nearest_label]
        energy = self._energy(feature)
        accepted = bool(distance <= radius)
        if accepted:
            outcome = self.label_outcomes[nearest_label]
            label = nearest_label
        else:
            outcome = SemanticOutcome.UNKNOWN_UAV_CLUSTER
            label = "unknown"
        return Prediction(
            outcome=outcome,
            label=label,
            energy=energy,
            distance=float(distance),
            nearest_label=nearest_label,
            accepted=accepted,
        )

    def predict_many(self, segments: list[Segment]) -> list[Prediction]:
        return [self.predict(segment) for segment in segments]

    def _nearest_distance(self, feature: np.ndarray) -> tuple[str, float]:
        distances = {
            label: float(np.linalg.norm(feature - prototype))
            for label, prototype in self.prototypes.items()
        }
        return min(distances.items(), key=lambda item: item[1])

    def _energy(self, feature: np.ndarray) -> float:
        distances = np.array(
            [np.linalg.norm(feature - prototype) for prototype in self.prototypes.values()],
            dtype=np.float64,
        )
        scaled = -distances / max(self.temperature, 1e-6)
        max_scaled = float(np.max(scaled))
        log_sum_exp = max_scaled + float(np.log(np.sum(np.exp(scaled - max_scaled))))
        return float(-self.temperature * log_sum_exp)

    def _scale(self, features: np.ndarray) -> np.ndarray:
        self._require_scaler()
        return (features - self.feature_mean) / self.feature_std

    def _require_scaler(self) -> None:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model has not been fit.")

    def _require_fit(self) -> None:
        self._require_scaler()
        if not self.prototypes:
            raise RuntimeError("Model has not been fit.")

