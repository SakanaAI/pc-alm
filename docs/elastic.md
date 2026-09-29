# Exploratory elastic dual coupling

A separate JAX harness (GeoDual), following this repository at
`660747f61a8a7e547c0ecd2c48c8883380a7d1f6`, tested whether symmetric multiplier
coupling can improve learning with short settling budgets. These are external
experimental results, not measurements produced by this repository's CLI or a
reproduction of the paper's one-epoch headline protocol. No training behavior
is changed by this note. The GeoDual harness is not included in this PR, so the
attached per-seed measurements are auditable summaries, not a runnable reproduction.

## Update

For hidden constraint layers indexed `i = 0,...,H-1`, pair `i` with `H-1-i`.
After the usual activity update, use the new activity residual and old duals:

```text
lambda_i[t+1] = lambda_i[t] + alpha * residual_i(z[t+1])
               - k[t] * (lambda_i[t] - lambda_(H-1-i)[t])
k[t] = 0.05 * max(1 - t / (floor(T/2) - 1), 0)
```

Here `alpha=1`, with one activity update per dual update. Both partners update
simultaneously. For `T=16/32/64`, coupling reaches zero at iteration `7/15/31`
(zero based); the middle layer is unchanged. Longer budgets also extend the
coupling phase. The original pre-final-dual weight-credit timing is retained.
This adds vector operations, not additional gradient evaluations or parameters.

The tested **elastic GDI** variant also initializes multipliers from a FIFO
memory: capacity 512, four same-label neighbors, original-image self-exclusion,
maximum age eight batches, strength 0.5. Retrieval ranks normalized hidden-state
and output-error similarity; temperature 0.1, hidden-similarity confidence
threshold 0.5, and exponential age attenuation weight the stored duals.
Thus the PC-ALM comparison measures **retrieval plus coupling**, not coupling alone.

## Matched results

Residual ReLU MLP, width/depth 32/32; original squared-error loss; batch 64;
Adam learning rate 0.001; activity rate 0.23588. Each dataset uses a fixed
stratified 55,000/5,000/10,000 train/validation/test split (split seed 42),
training seeds 0/1/2, and validation-selected checkpoints. Tests are evaluated
only after training. Coupling strength was fixed before these comparisons.

Mean held-out accuracy (%), **five → ten epochs**:

| Dataset | Settling steps | PC-ALM | Elastic GDI |
|---|---:|---:|---:|
| MNIST | 16 | 74.30 → 78.74 | 91.43 → 92.74 |
| MNIST | 32 | 77.89 → 81.31 | 92.07 → 93.21 |
| MNIST | 64 | 92.24 → 93.66 | 92.37 → 93.69 |
| Fashion-MNIST | 16 | 71.84 → 74.34 | 81.92 → 83.23 |
| Fashion-MNIST | 32 | 74.01 → 76.01 | 82.31 → 83.48 |
| Fashion-MNIST | 64 | 83.01 → 84.59 | 83.06 → 84.35 |

[Per-seed results](elastic.csv) include validation-selected epochs and training
times. The 36 ten-epoch comparisons reuse six previously completed elastic-GDI16
runs; the other 30 were newly trained. All first-five validation-accuracy
trajectories matched the corresponding five-epoch runs, and saved weights were
finite. Timings are serial local CPU seconds including compilation, excluding
loading and evaluation; reused timings are marked explicitly.

An earlier ten-epoch, 16-step comparison against **ordinary GDI** helps separate
the contribution of coupling: MNIST improved from 89.69% to 92.74%, and Fashion
from 80.46% to 83.23%, with all six paired seeds improving. Those controls are
also included in the CSV; they use the same retrieval settings without coupling.

The observed benefit is at short settling budgets. At 64 steps and ten epochs,
MNIST is effectively tied (mixed paired differences), while Fashion favors
PC-ALM by 0.24 points in all three seeds. This pilot does not establish statistical
general superiority, improved convergence guarantees, or ImageNet performance.
