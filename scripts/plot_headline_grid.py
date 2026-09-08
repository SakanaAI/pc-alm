from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PC-ALM minus PC grid from cells.csv.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/headline_grid/gain_pcalm_minus_pc.png"))
    parser.add_argument("--activation", default="relu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    sub = [r for r in rows if r["activation"] == args.activation]
    widths = sorted({int(r["width"]) for r in sub})
    depths = sorted({int(r["depth"]) for r in sub}, reverse=True)
    gain = np.full((len(depths), len(widths)), np.nan)
    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            pc = mean_acc(sub, "pc", width, depth)
            alm = mean_acc(sub, "pcalm", width, depth)
            if pc is not None and alm is not None:
                gain[i, j] = alm - pc
    vmax = np.nanmax(np.abs(gain)) if np.isfinite(gain).any() else 1.0
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(gain, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(widths)), widths)
    ax.set_yticks(range(len(depths)), depths)
    ax.set_xlabel("width N")
    ax.set_ylabel("depth L")
    ax.set_title(f"PC-ALM - PC ({args.activation})")
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            if np.isfinite(gain[i, j]):
                ax.text(j, i, f"{gain[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="test accuracy gain")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_acc(rows, method: str, width: int, depth: int) -> float | None:
    vals = [
        float(r["final_test_acc"])
        for r in rows
        if r["method"] == method and int(r["width"]) == width and int(r["depth"]) == depth
    ]
    if not vals:
        return None
    return float(np.mean(vals))


if __name__ == "__main__":
    main()
