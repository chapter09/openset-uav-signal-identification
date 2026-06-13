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
