# Open-Set UAV Signal Identification Simulator

This repository contains a first simulation environment for open-set UAV RF signal identification. It generates complex I/Q segments with receiver metadata and semantic labels:

- known UAV ID
- known non-UAV emitter or interference
- true background/noise
- unknown UAV cluster

The design is aligned with the two papers you referenced:

- Ma et al. 2025: prototype-style feature geometry and adaptive distance thresholds.
- Long et al. 2026: I/Q UAV segments, SNR/openness stress tests, learnable embedding geometry, and energy-style rejection.

The simulator itself is model-agnostic. It provides the labels, metadata, and scene structure needed to train or evaluate closed-set and open-set methods.

## Quick Start

```bash
python -m pip install -e .
openset-uav-sim generate --output data/demo --seed 7
```

If you are using the bundled Codex Python runtime in this workspace:

```bash
PYTHONPATH=src /Users/haowang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m openset_uav_sim.cli generate --output data/demo --seed 7
```

Outputs are written as compressed NumPy arrays plus JSONL metadata:

```text
data/demo/
  train.npz
  train.jsonl
  val.npz
  val.jsonl
  test.npz
  test.jsonl
  summary.json
```

## Python Example

```python
from openset_uav_sim import OpenSetUAVEnvironment, PrototypeOpenSetModel

env = OpenSetUAVEnvironment.default(seed=2026)
splits = env.make_open_set_splits(train_per_known=24, test_per_known=12, unknown_per_cluster=12)

model = PrototypeOpenSetModel(tail_quantile=0.95)
model.fit(splits["train"])

prediction = model.predict(splits["test"][0])
print(prediction.outcome, prediction.label, prediction.energy)
```

## GE-OSR Reproduction

The paper implementation is reproduced in `openset_uav_sim.geosr` as an optional PyTorch module:

- Frequency-Conditioned Temporal Modulation (FCTM)
- parallel Convolutional-Transformer hybrid blocks (CTBlocks)
- learnable unit-normalized class embeddings
- Dual-Constraint Embedding Loss (DCEL)
- Free Energy Alignment Loss (FEAL)
- EMA adaptive energy thresholding

Install the optional training dependency:

```bash
python -m pip install -e '.[torch]'
```

Train GE-OSR directly against simulated open-set splits:

```bash
openset-uav-sim train-geosr --epochs 20 --seed 2026
```

The default GE-OSR hyperparameters follow Table 1 of Long et al. 2026:

```text
alpha=32, delta=0.1, T=10.0, E0=-0.1, lambda1=0.3,
lambda2=1.0, beta=0.2, batch_size=128, optimizer=AdamW,
learning_rate=0.001
```

In the original paper, only known UAV classes are trained and unknown UAV classes are withheld. In this simulator, the same open-set rule is preserved while the known semantic set is expanded to include known UAV IDs, known non-UAV emitters, and background/noise.

## CageDroneRF Integration

The simulator can import real CageDroneRF/U-RAPTOR raw recordings into the same `Segment` format used by synthetic data and GE-OSR. The public toolkit describes CageDroneRF raw recordings as `.dat` files containing `complex64` I/Q samples, with filename metadata such as manufacturer, model, bandwidth, center frequency, and operation mode.

After you receive/download the CageDroneRF data, convert it into simulator splits:

```bash
openset-uav-sim import-cagedronerf \
  --raw-root /path/to/CageDroneRF/raw \
  --output data/cagedronerf \
  --unknown-label DJI_Mavic3 \
  --unknown-label Autel_EVO \
  --max-segments-per-label 200
```

If you used their processing script and have a `meta_data.json`, pass it too:

```bash
openset-uav-sim import-cagedronerf \
  --raw-root /path/to/CageDroneRF/raw \
  --metadata /path/to/CageDroneRF/processed/meta_data.json \
  --output data/cagedronerf
```

Mapping rules:

- Drone labels become `known_uav_id`.
- Labels passed with `--unknown-label` become `unknown_uav_cluster` and are withheld from train/validation splits.
- `non-drone/` recordings become `known_non_uav_emitter`.
- Labels or modes containing `NoDrone`, `background`, or `noise` become `true_background_noise`.

To train GE-OSR on CageDroneRF and write evaluation tables/figures:

```bash
openset-uav-sim evaluate-geosr-cagedronerf \
  --raw-root /path/to/CageDroneRF/raw \
  --report-dir reports/geosr-cagedronerf \
  --unknown-label DJI_Mavic3 \
  --unknown-label Autel_EVO \
  --epochs 20 \
  --max-segments-per-label 200
```

The report directory contains:

```text
metrics_summary.md
metrics_summary.csv
per_label_metrics.md
per_label_metrics.csv
metrics.json
roc_curve.svg
oscr_curve.svg
energy_histogram.svg
confusion_matrix.svg
training_history.json
run_config.json
```

## Core Assumptions

Each segment is a complex baseband I/Q vector with metadata:

```text
center_frequency_hz, bandwidth_hz, timestamp_s, receiver_id,
gain_db, location, antenna, estimated_snr_db
```

Known UAVs and known non-UAV emitters are available during training. Unknown UAV clusters are withheld from training and appear in validation/test depending on the split policy. Unknown clusters carry stable `cluster_id` values so they can later be renamed as a new UAV, model, controller, party, or other semantic identity after labeling.

## Run Tests

```bash
PYTHONPATH=src /Users/haowang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests
```
