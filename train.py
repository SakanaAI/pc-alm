from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pcalm.config import ExperimentConfig, load_config
from pcalm.training import train_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual MLP with BP, PC, or PC-ALM.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "fashion_mnist"])
    parser.add_argument("--method", choices=["bp", "pc", "pcalm"])
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--state-lr", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eta0", type=float)
    parser.add_argument("--gamma0", type=float, help="Fixed at 1 in this reference implementation.")
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--test-subset", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--data-dir", type=str, default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else ExperimentConfig()
    if args.dataset is not None:
        config = replace(config, dataset=args.dataset)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    model_updates = {
        "width": args.width,
        "depth": args.depth,
        "activation": args.activation,
    }
    method_updates = {
        "name": args.method,
        "budget": args.budget,
        "alpha": args.alpha,
        "state_lr": args.state_lr,
        "rho": args.rho,
    }
    training_updates = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eta0": args.eta0,
        "gamma0": args.gamma0,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
    }
    config = replace(
        config,
        model=replace(config.model, **{k: v for k, v in model_updates.items() if v is not None}),
        method=replace(config.method, **{k: v for k, v in method_updates.items() if v is not None}),
        training=replace(config.training, **{k: v for k, v in training_updates.items() if v is not None}),
    )
    summary = train_one(config, data_dir=args.data_dir)
    print(summary)


if __name__ == "__main__":
    main()
