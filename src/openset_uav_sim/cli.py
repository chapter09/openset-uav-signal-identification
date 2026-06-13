from __future__ import annotations

import argparse
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

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

