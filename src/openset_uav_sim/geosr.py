from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .prototype import Prediction
from .types import Segment, SemanticOutcome

try:  # Optional: the simulator itself stays usable without PyTorch.
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised by import-time tests.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


TORCH_AVAILABLE = torch is not None


class TorchUnavailableError(ImportError):
    """Raised when the optional GE-OSR PyTorch implementation is requested."""


def require_torch() -> None:
    if torch is None:
        raise TorchUnavailableError(
            "GE-OSR requires PyTorch. Install it with `python -m pip install -e '.[torch]'` "
            "or install a platform-specific torch build from pytorch.org."
        )


@dataclass(frozen=True)
class GEOSRConfig:
    """GE-OSR hyperparameters.

    The defaults follow Table 1 of Long et al. 2026 where specified:
    alpha=32, delta=0.1, T=10.0, E0=-0.1, lambda1=0.3,
    lambda2=1.0, beta=0.2, batch size=128, AdamW, learning rate=0.001.
    """

    alpha: float = 32.0
    delta: float = 0.1
    temperature: float = 10.0
    target_energy: float = -0.1
    lambda_dce: float = 0.3
    lambda_fea: float = 1.0
    beta: float = 0.2
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 20
    ema_rate: float = 0.05
    input_channels: int = 2
    base_channels: int = 64
    stage_channels: tuple[int, ...] = (64, 128, 128)
    feature_dim: int = 128
    transformer_heads: int = 4
    transformer_ff_multiplier: int = 4
    conv_expansion: int = 2
    conv_kernel_size: int = 15
    dropout: float = 0.1
    num_workers: int = 0


@dataclass(frozen=True)
class TrainingEpoch:
    epoch: int
    loss: float
    dce_loss: float
    fea_loss: float
    threshold: float


@dataclass(frozen=True)
class GEOSRMetrics:
    known_accuracy: float
    unknown_rejection_rate: float
    macro_accuracy: float
    known_count: int
    unknown_count: int


def build_label_index(segments: list[Segment]) -> dict[str, int]:
    labels = sorted({segment.training_label for segment in segments if not segment.is_unknown})
    return {label: index for index, label in enumerate(labels)}


def build_outcome_index(segments: list[Segment]) -> dict[str, SemanticOutcome]:
    outcomes: dict[str, SemanticOutcome] = {}
    for segment in segments:
        if not segment.is_unknown:
            outcomes.setdefault(segment.training_label, segment.outcome)
    return outcomes


def segment_to_tensor(segment: Segment) -> Any:
    require_torch()
    real = np.real(segment.iq).astype(np.float32)
    imag = np.imag(segment.iq).astype(np.float32)
    return torch.from_numpy(np.stack([real, imag], axis=0))


if TORCH_AVAILABLE:

    class SegmentDataset(Dataset):
        """Torch dataset adapter for simulator segments.

        Unknown UAV clusters are excluded for training, matching the paper's open-set
        protocol. Background/noise and known non-UAV emitters remain valid known
        semantic classes in this simulator.
        """

        def __init__(
            self,
            segments: list[Segment],
            label_to_index: dict[str, int] | None = None,
            include_unknown: bool = False,
        ) -> None:
            self.label_to_index = label_to_index or build_label_index(segments)
            self.items: list[Segment] = []
            for segment in segments:
                if segment.is_unknown and not include_unknown:
                    continue
                if segment.training_label in self.label_to_index:
                    self.items.append(segment)

            if not self.items:
                raise ValueError("SegmentDataset has no usable segments.")

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            segment = self.items[index]
            x = segment_to_tensor(segment)
            y = torch.tensor(self.label_to_index[segment.training_label], dtype=torch.long)
            return x, y


    class FrequencyConditionedTemporalModulation(nn.Module):
        """FCTM module from GE-OSR."""

        def __init__(self, input_channels: int, hidden_channels: int) -> None:
            super().__init__()
            self.temporal_path = nn.Sequential(
                nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
            )
            self.frequency_path = nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.gamma = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.beta = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

        def forward(self, x: Any) -> Any:
            temporal = self.temporal_path(x)
            complex_iq = torch.complex(x[:, 0], x[:, 1])
            spectrum = torch.fft.fftshift(torch.fft.fft(complex_iq, dim=-1), dim=-1)
            spectrum = torch.log1p(torch.abs(spectrum)).unsqueeze(1)
            descriptor = self.frequency_path(spectrum).squeeze(-1)
            gamma = (1.0 + torch.tanh(self.gamma(descriptor))).unsqueeze(-1)
            beta = self.beta(descriptor).unsqueeze(-1)
            return gamma * temporal + beta


    class CTBlock(nn.Module):
        """Parallel convolutional-Transformer hybrid block from GE-OSR."""

        def __init__(self, channels: int, config: GEOSRConfig) -> None:
            super().__init__()
            expanded = channels * config.conv_expansion
            self.conv_branch = nn.Sequential(
                nn.Conv1d(channels, expanded, kernel_size=1, bias=False),
                nn.BatchNorm1d(expanded),
                nn.GELU(),
                nn.Conv1d(
                    expanded,
                    expanded,
                    kernel_size=config.conv_kernel_size,
                    padding=config.conv_kernel_size // 2,
                    groups=expanded,
                    bias=False,
                ),
                nn.BatchNorm1d(expanded),
                nn.GELU(),
                nn.Conv1d(expanded, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=config.transformer_heads,
                dim_feedforward=channels * config.transformer_ff_multiplier,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_branch = nn.TransformerEncoder(layer, num_layers=1)
            self.gate_logit = nn.Parameter(torch.tensor(0.0))
            self.dropout = nn.Dropout(config.dropout)
            self.output_norm = nn.BatchNorm1d(channels)

        def forward(self, x: Any) -> Any:
            local = self.conv_branch(x)
            global_context = self.transformer_branch(x.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(self.gate_logit)
            fused = gate * local + (1.0 - gate) * global_context
            return self.output_norm(x + self.dropout(fused))


    class TimeFrequencyHybridExtractor(nn.Module):
        """FCTM plus stacked CTBlocks with temporal downsampling."""

        def __init__(self, config: GEOSRConfig) -> None:
            super().__init__()
            if not config.stage_channels:
                raise ValueError("stage_channels must contain at least one channel width.")
            self.fctm = FrequencyConditionedTemporalModulation(
                input_channels=config.input_channels,
                hidden_channels=config.stage_channels[0],
            )
            stages: list[nn.Module] = []
            channels = config.stage_channels[0]
            stages.append(CTBlock(channels, config))
            for next_channels in config.stage_channels[1:]:
                stages.append(
                    nn.Sequential(
                        nn.Conv1d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm1d(next_channels),
                        nn.GELU(),
                    )
                )
                channels = next_channels
                stages.append(CTBlock(channels, config))
            self.stages = nn.Sequential(*stages)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.projection = nn.Sequential(
                nn.Linear(channels, config.feature_dim),
                nn.LayerNorm(config.feature_dim),
            )

        def forward(self, x: Any) -> Any:
            features = self.fctm(x)
            features = self.stages(features)
            pooled = self.pool(features).squeeze(-1)
            return self.projection(pooled)


    class GEOSRModel(nn.Module):
        """Geometry-energy open-set recognition model."""

        def __init__(self, num_classes: int, config: GEOSRConfig | None = None) -> None:
            super().__init__()
            if num_classes <= 0:
                raise ValueError("num_classes must be positive.")
            self.config = config or GEOSRConfig()
            self.extractor = TimeFrequencyHybridExtractor(self.config)
            embeddings = torch.randn(num_classes, self.config.feature_dim)
            embeddings = F.normalize(embeddings, dim=1)
            self.class_embeddings = nn.Parameter(embeddings)
            self.register_buffer("energy_mean", torch.tensor(self.config.target_energy, dtype=torch.float32))
            self.register_buffer("energy_var", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("threshold_ready", torch.tensor(False, dtype=torch.bool))

        def encode(self, x: Any) -> Any:
            return F.normalize(self.extractor(x), dim=1)

        def normalized_embeddings(self) -> Any:
            return F.normalize(self.class_embeddings, dim=1)

        def similarities(self, z: Any) -> Any:
            return z @ self.normalized_embeddings().transpose(0, 1)

        def energy_from_similarities(self, similarities: Any) -> Any:
            distances = 1.0 - similarities
            return -(1.0 / self.config.temperature) * torch.logsumexp(
                -self.config.temperature * distances,
                dim=1,
            )

        def forward(self, x: Any) -> dict[str, Any]:
            z = self.encode(x)
            similarities = self.similarities(z)
            energy = self.energy_from_similarities(similarities)
            return {"features": z, "similarities": similarities, "energy": energy}

        def dce_loss(self, similarities: Any, labels: Any) -> Any:
            batch_size, class_count = similarities.shape
            true_sim = similarities.gather(1, labels.view(-1, 1)).squeeze(1)
            intra = F.softplus(-self.config.alpha * (true_sim - self.config.delta)).mean()

            if class_count == 1:
                inter = similarities.new_tensor(0.0)
            else:
                mask = torch.zeros_like(similarities, dtype=torch.bool)
                mask.scatter_(1, labels.view(-1, 1), True)
                wrong_sim = similarities.masked_select(~mask)
                inter = F.softplus(self.config.alpha * (wrong_sim + self.config.delta)).mean()

            return intra + inter

        def loss(self, x: Any, labels: Any) -> dict[str, Any]:
            outputs = self(x)
            dce = self.dce_loss(outputs["similarities"], labels)
            fea = ((outputs["energy"] - self.config.target_energy) ** 2).mean()
            total = self.config.lambda_dce * dce + self.config.lambda_fea * fea
            return {"loss": total, "dce_loss": dce, "fea_loss": fea, **outputs}

        @torch.no_grad()
        def update_threshold(self, energy: Any) -> None:
            batch_mean = energy.detach().mean()
            batch_var = energy.detach().var(unbiased=False)
            if not bool(self.threshold_ready.item()):
                self.energy_mean.copy_(batch_mean)
                self.energy_var.copy_(batch_var.clamp_min(1e-8))
                self.threshold_ready.copy_(torch.tensor(True, device=self.threshold_ready.device))
                return
            eta = self.config.ema_rate
            self.energy_mean.mul_(1.0 - eta).add_(eta * batch_mean)
            self.energy_var.mul_(1.0 - eta).add_(eta * batch_var).clamp_(min=1e-8)

        @property
        def threshold(self) -> Any:
            return self.energy_mean + self.config.beta * torch.sqrt(self.energy_var.clamp_min(1e-8))

        @torch.no_grad()
        def predict_batch(self, x: Any) -> dict[str, Any]:
            outputs = self(x)
            similarities = outputs["similarities"]
            energy = outputs["energy"]
            nearest_similarity, nearest_index = similarities.max(dim=1)
            accepted = energy <= self.threshold
            return {
                "class_index": nearest_index,
                "accepted": accepted,
                "energy": energy,
                "nearest_similarity": nearest_similarity,
                "distance": 1.0 - nearest_similarity,
            }


    @dataclass
    class GEOSRTrainer:
        model: GEOSRModel
        label_to_index: dict[str, int]
        outcome_by_label: dict[str, SemanticOutcome]
        config: GEOSRConfig = field(default_factory=GEOSRConfig)
        device: str | None = None

        @classmethod
        def from_segments(
            cls,
            train_segments: list[Segment],
            config: GEOSRConfig | None = None,
            device: str | None = None,
        ) -> "GEOSRTrainer":
            cfg = config or GEOSRConfig()
            label_to_index = build_label_index(train_segments)
            if not label_to_index:
                raise ValueError("Training segments must include at least one known class.")
            outcome_by_label = build_outcome_index(train_segments)
            model = GEOSRModel(num_classes=len(label_to_index), config=cfg)
            return cls(model=model, label_to_index=label_to_index, outcome_by_label=outcome_by_label, config=cfg, device=device)

        @property
        def index_to_label(self) -> dict[int, str]:
            return {index: label for label, index in self.label_to_index.items()}

        def fit(self, train_segments: list[Segment], epochs: int | None = None) -> list[TrainingEpoch]:
            device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.to(device)
            dataset = SegmentDataset(train_segments, label_to_index=self.label_to_index)
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            history: list[TrainingEpoch] = []
            total_epochs = epochs or self.config.epochs
            for epoch in range(1, total_epochs + 1):
                self.model.train()
                running = {"loss": 0.0, "dce_loss": 0.0, "fea_loss": 0.0}
                seen = 0
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = self.model.loss(x, y)
                    losses["loss"].backward()
                    optimizer.step()
                    self.model.update_threshold(losses["energy"].detach())
                    batch_size = int(x.shape[0])
                    seen += batch_size
                    for key in running:
                        running[key] += float(losses[key].detach().cpu()) * batch_size
                history.append(
                    TrainingEpoch(
                        epoch=epoch,
                        loss=running["loss"] / max(seen, 1),
                        dce_loss=running["dce_loss"] / max(seen, 1),
                        fea_loss=running["fea_loss"] / max(seen, 1),
                        threshold=float(self.model.threshold.detach().cpu()),
                    )
                )
            return history

        @torch.no_grad()
        def predict_segment(self, segment: Segment) -> Prediction:
            device = next(self.model.parameters()).device
            self.model.eval()
            x = segment_to_tensor(segment).unsqueeze(0).to(device)
            output = self.model.predict_batch(x)
            index = int(output["class_index"][0].detach().cpu())
            nearest_label = self.index_to_label[index]
            accepted = bool(output["accepted"][0].detach().cpu())
            if accepted:
                outcome = self.outcome_by_label[nearest_label]
                label = nearest_label
            else:
                outcome = SemanticOutcome.UNKNOWN_UAV_CLUSTER
                label = "unknown"
            return Prediction(
                outcome=outcome,
                label=label,
                energy=float(output["energy"][0].detach().cpu()),
                distance=float(output["distance"][0].detach().cpu()),
                nearest_label=nearest_label,
                accepted=accepted,
            )

        def predict_many(self, segments: list[Segment]) -> list[Prediction]:
            return [self.predict_segment(segment) for segment in segments]


    def evaluate_geosr(trainer: GEOSRTrainer, segments: list[Segment]) -> GEOSRMetrics:
        predictions = trainer.predict_many(segments)
        known_total = 0
        known_correct = 0
        unknown_total = 0
        unknown_rejected = 0
        for segment, prediction in zip(segments, predictions):
            if segment.is_unknown:
                unknown_total += 1
                if prediction.outcome == SemanticOutcome.UNKNOWN_UAV_CLUSTER:
                    unknown_rejected += 1
            else:
                known_total += 1
                if prediction.accepted and prediction.label == segment.training_label:
                    known_correct += 1
        known_accuracy = known_correct / known_total if known_total else 0.0
        rejection_rate = unknown_rejected / unknown_total if unknown_total else 0.0
        macro_accuracy = (known_accuracy + rejection_rate) / 2.0 if unknown_total else known_accuracy
        return GEOSRMetrics(
            known_accuracy=known_accuracy,
            unknown_rejection_rate=rejection_rate,
            macro_accuracy=macro_accuracy,
            known_count=known_total,
            unknown_count=unknown_total,
        )

else:

    class SegmentDataset(Dataset):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class FrequencyConditionedTemporalModulation:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class CTBlock:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class TimeFrequencyHybridExtractor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class GEOSRModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class GEOSRTrainer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    def evaluate_geosr(*args: Any, **kwargs: Any) -> GEOSRMetrics:
        require_torch()
        raise AssertionError("unreachable")

