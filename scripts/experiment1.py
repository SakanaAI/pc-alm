from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcalm.config import load_config
from pcalm.training import train_one


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small BP/PC/PC-ALM grid.")
    parser.add_argument("--config", type=Path, default=Path("configs/headline_fashion.yaml"))
    parser.add_argument("--widths", default="8,16,32")
    parser.add_argument("--depths", default="8,16,32")
    parser.add_argument("--activations", default="linear,tanh,relu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="bp,pc,pcalm")
    parser.add_argument("--budget-rule", choices=["L", "2L"], default="2L")
    parser.add_argument("--state-lr-depth-table", type=Path, default=Path("configs/eta_by_depth.csv"))
    parser.add_argument("--state-lr-table", type=Path, help="Optional per-cell eta table (e.g. configs/eta_best_by_cell.csv); overrides depth table.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/headline_grid"))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--quick", action="store_true", help="Use tiny synthetic-sized subsets from the config for a smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    widths = parse_csv_ints(args.widths)
    depths = parse_csv_ints(args.depths)
    activations = parse_csv_strings(args.activations)
    seeds = parse_csv_ints(args.seeds)
    methods = parse_csv_strings(args.methods)
    state_lrs = load_state_lr_table(args.state_lr_table) if args.state_lr_table else {}
    depth_state_lrs = load_depth_state_lr_table(args.state_lr_depth_table)
    rows = []
    for activation in activations:
        for width in widths:
            for depth in depths:
                budget = depth if args.budget_rule == "L" else 2 * depth
                state_lr = state_lrs.get(
                    (base.dataset, activation, width, depth),
                    depth_state_lrs.get(depth, base.method.state_lr),
                )
                for seed in seeds:
                    for method in methods:
                        run_dir = args.output_dir / f"{base.dataset}_{activation}_n{width}_l{depth}_seed{seed}_{method}"
                        cfg = replace(
                            base,
                            output_dir=str(run_dir),
                            model=replace(base.model, width=width, depth=depth, activation=activation),
                            method=replace(base.method, name=method, budget=budget if method != "bp" else 0, state_lr=state_lr),
                            training=replace(base.training, seed=seed),
                        )
                        if args.quick:
                            cfg = replace(
                                cfg,
                                dataset="synthetic",
                                training=replace(cfg.training, train_subset=128, test_subset=64),
                            )
                        summary = train_one(cfg, data_dir=args.data_dir)
                        rows.append(summary)
                        print(summary, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cells.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_state_lr_table(path: Path) -> dict[tuple[str, str, int, int], float]:
    out: dict[tuple[str, str, int, int], float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["activation"], int(row["N"]), int(row["L"]))
            out[key] = float(row["eta_best_1_over_lambda_median"])
    return out


def load_depth_state_lr_table(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["L"])] = float(row["state_lr"])
    return out


if __name__ == "__main__":
    main()
