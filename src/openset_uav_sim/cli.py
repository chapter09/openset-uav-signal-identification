from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .environment import OpenSetUAVEnvironment
from .export import save_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate open-set UAV RF simulation datasets.")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Generate train/val/test open-set splits.")
    generate.add_argument("--output", required=True, help="Output directory.")
    generate.add_argument("--seed", type=int, default=2026, help="Random seed.")
    generate.add_argument("--train-per-known", type=int, default=48)
    generate.add_argument("--val-per-known", type=int, default=16)
    generate.add_argument("--test-per-known", type=int, default=24)
    generate.add_argument("--unknown-per-cluster", type=int, default=24)
    generate.add_argument("--background-per-split", type=int, default=48)

    train = subparsers.add_parser("train-geosr", help="Train the GE-OSR reproduction on simulated splits.")
    train.add_argument("--seed", type=int, default=2026, help="Random seed.")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--train-per-known", type=int, default=48)
    train.add_argument("--val-per-known", type=int, default=16)
    train.add_argument("--test-per-known", type=int, default=24)
    train.add_argument("--unknown-per-cluster", type=int, default=24)
    train.add_argument("--background-per-split", type=int, default=48)
    train.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")

    cage = subparsers.add_parser("import-cagedronerf", help="Convert CageDroneRF raw .dat files to simulator splits.")
    cage.add_argument("--raw-root", required=True, help="Directory containing CageDroneRF .dat recordings.")
    cage.add_argument("--metadata", default=None, help="Optional CageDroneRF meta_data.json or JSONL file.")
    cage.add_argument("--output", required=True, help="Output directory for simulator NPZ/JSONL splits.")
    cage.add_argument("--known-label", action="append", default=[], help="Label to force as known. Repeatable.")
    cage.add_argument("--unknown-label", action="append", default=[], help="Drone label to withhold as unknown. Repeatable.")
    cage.add_argument("--segment-length", type=int, default=4096)
    cage.add_argument("--stride", type=int, default=4096)
    cage.add_argument("--max-segments-per-recording", type=int, default=None)
    cage.add_argument("--max-segments-per-label", type=int, default=None)
    cage.add_argument("--seed", type=int, default=2026)

    eval_cage = subparsers.add_parser(
        "evaluate-geosr-cagedronerf",
        help="Train GE-OSR on CageDroneRF splits and save metrics tables and SVG figures.",
    )
    eval_cage.add_argument("--raw-root", required=True, help="Directory containing CageDroneRF .dat recordings.")
    eval_cage.add_argument("--metadata", default=None, help="Optional CageDroneRF meta_data.json or JSONL file.")
    eval_cage.add_argument("--report-dir", required=True, help="Directory for metrics tables and SVG figures.")
    eval_cage.add_argument("--known-label", action="append", default=[], help="Label to force as known. Repeatable.")
    eval_cage.add_argument("--unknown-label", action="append", default=[], help="Drone label to withhold as unknown. Repeatable.")
    eval_cage.add_argument("--segment-length", type=int, default=4096)
    eval_cage.add_argument("--stride", type=int, default=4096)
    eval_cage.add_argument("--max-segments-per-recording", type=int, default=None)
    eval_cage.add_argument("--max-segments-per-label", type=int, default=None)
    eval_cage.add_argument("--epochs", type=int, default=None)
    eval_cage.add_argument("--batch-size", type=int, default=None)
    eval_cage.add_argument("--learning-rate", type=float, default=None)
    eval_cage.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    eval_cage.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        env = OpenSetUAVEnvironment.default(seed=args.seed)
        splits = env.make_open_set_splits(
            train_per_known=args.train_per_known,
            val_per_known=args.val_per_known,
            test_per_known=args.test_per_known,
            unknown_per_cluster=args.unknown_per_cluster,
            background_per_split=args.background_per_split,
        )
        summary = save_splits(splits, args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "train-geosr":
        from .geosr import GEOSRConfig, GEOSRTrainer, evaluate_geosr, require_torch

        try:
            require_torch()
        except ImportError as exc:
            parser.error(str(exc))

        env = OpenSetUAVEnvironment.default(seed=args.seed)
        splits = env.make_open_set_splits(
            train_per_known=args.train_per_known,
            val_per_known=args.val_per_known,
            test_per_known=args.test_per_known,
            unknown_per_cluster=args.unknown_per_cluster,
            background_per_split=args.background_per_split,
        )
        config = GEOSRConfig(epochs=args.epochs or GEOSRConfig().epochs)
        trainer = GEOSRTrainer.from_segments(splits["train"], config=config, device=args.device)
        history = trainer.fit(splits["train"], epochs=args.epochs)
        metrics = evaluate_geosr(trainer, splits["test"])
        print(
            json.dumps(
                {
                    "config": asdict(config),
                    "labels": trainer.label_to_index,
                    "history": [asdict(item) for item in history],
                    "test_metrics": asdict(metrics),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "import-cagedronerf":
        from .cagedronerf import CageDroneRFConfig, CageDroneRFLoader

        config = CageDroneRFConfig(
            raw_root=args.raw_root,
            metadata_path=args.metadata,
            segment_length=args.segment_length,
            stride=args.stride,
        )
        loader = CageDroneRFLoader(
            config=config,
            known_labels=args.known_label,
            unknown_labels=args.unknown_label,
        )
        splits = loader.make_open_set_splits(
            seed=args.seed,
            max_segments_per_recording=args.max_segments_per_recording,
            max_segments_per_label=args.max_segments_per_label,
        )
        summary = save_splits(splits, args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "evaluate-geosr-cagedronerf":
        from pathlib import Path

        from .cagedronerf import CageDroneRFConfig, CageDroneRFLoader
        from .evaluation import evaluate_predictions, save_open_set_report
        from .geosr import GEOSRConfig, GEOSRTrainer, require_torch

        try:
            require_torch()
        except ImportError as exc:
            parser.error(str(exc))

        cage_config = CageDroneRFConfig(
            raw_root=args.raw_root,
            metadata_path=args.metadata,
            segment_length=args.segment_length,
            stride=args.stride,
        )
        loader = CageDroneRFLoader(
            config=cage_config,
            known_labels=args.known_label,
            unknown_labels=args.unknown_label,
        )
        splits = loader.make_open_set_splits(
            seed=args.seed,
            max_segments_per_recording=args.max_segments_per_recording,
            max_segments_per_label=args.max_segments_per_label,
        )
        geosr_config = GEOSRConfig(
            epochs=args.epochs or GEOSRConfig().epochs,
            batch_size=args.batch_size or GEOSRConfig().batch_size,
            learning_rate=args.learning_rate or GEOSRConfig().learning_rate,
        )
        trainer = GEOSRTrainer.from_segments(splits["train"], config=geosr_config, device=args.device)
        history = trainer.fit(splits["train"], epochs=args.epochs)
        predictions = trainer.predict_many(splits["test"])
        report = evaluate_predictions(splits["test"], predictions)

        report_dir = Path(args.report_dir)
        save_open_set_report(report, report_dir)
        with (report_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in history], handle, indent=2, sort_keys=True)
        with (report_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "geosr_config": asdict(geosr_config),
                    "labels": trainer.label_to_index,
                    "split_counts": {name: len(items) for name, items in splits.items()},
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(json.dumps(asdict(report.metrics), indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
