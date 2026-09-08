from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import Params, block_pred, forward


@dataclass(frozen=True)
class Schedule:
    family: str
    budget: int
    alpha: float = 0.0
    inner_steps: int = 1
    weight_credit_timing: str = "pre_dual_energy"


def supervised_loss(params: Params, scales, skips, x, y, free, phi) -> jax.Array:
    y_pred = block_pred(params[-1], scales[-1], skips[-1], free[-1], phi, is_first=False)
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def bp_loss(params: Params, scales, skips, x, y, phi) -> jax.Array:
    y_pred = forward(params, scales, skips, x, phi)[-1]
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def free_init(params: Params, scales, skips, x, phi) -> list[jax.Array]:
    return forward(params, scales, skips, x, phi)[:-1]


def constraint_residuals(params: Params, scales, skips, x, free, phi) -> list[jax.Array]:
    """Hidden model-edge constraints only; the label residual is not constrained."""
    residuals = []
    for layer_ix, z_l in enumerate(free):
        z_prev = x if layer_ix == 0 else free[layer_ix - 1]
        pred = block_pred(
            params[layer_ix],
            scales[layer_ix],
            skips[layer_ix],
            z_prev,
            phi,
            is_first=(layer_ix == 0),
        )
        residuals.append(z_l - pred)
    return residuals


def zero_duals_like(residuals: list[jax.Array]) -> list[jax.Array]:
    return [jnp.zeros_like(c) for c in residuals]


def al_energy_shifted(params: Params, scales, skips, x, y, free, duals, rho: float, phi) -> jax.Array:
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    total = supervised_loss(params, scales, skips, x, y, free, phi)
    batch_size = x.shape[0]
    for residual, dual in zip(residuals, duals):
        shifted = residual + dual / rho
        total = total + 0.5 * rho * jnp.sum(shifted * shifted) / batch_size
    return total


def run_pc(params: Params, scales, skips, x, y, *, state_lr: float, rho: float, steps: int, phi):
    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    return _solve_inner(params, scales, skips, x, y, free0, duals0, state_lr, rho, steps, phi), duals0


def run_pcalm(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
    inner_steps: int,
    weight_credit_timing: str,
    phi,
):
    if budget < 1:
        raise ValueError("PC-ALM budget must be at least 1")
    if inner_steps < 1:
        raise ValueError("PC-ALM inner_steps must be at least 1")
    if weight_credit_timing not in {"pre_dual_energy", "post_dual_energy"}:
        raise ValueError("weight_credit_timing must be pre_dual_energy or post_dual_energy")

    free = free_init(params, scales, skips, x, phi)
    duals = zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi))

    def outer(carry, _):
        free_c, duals_c = carry
        free_c = _solve_inner(params, scales, skips, x, y, free_c, duals_c, state_lr, rho, inner_steps, phi)
        residuals = constraint_residuals(params, scales, skips, x, free_c, phi)
        duals_next = [lam + alpha * r for lam, r in zip(duals_c, residuals)]
        return (free_c, duals_next), None

    if budget > 1:
        (free, duals_before), _ = jax.lax.scan(outer, (free, duals), xs=None, length=budget - 1)
    else:
        duals_before = duals

    free = _solve_inner(params, scales, skips, x, y, free, duals_before, state_lr, rho, inner_steps, phi)
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    duals_after = [lam + alpha * r for lam, r in zip(duals_before, residuals)]
    duals_weight = duals_before if weight_credit_timing == "pre_dual_energy" else duals_after
    return free, duals_weight


def infer_for_schedule(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    if schedule.family == "pc":
        return run_pc(params, scales, skips, x, y, state_lr=state_lr, rho=rho, steps=schedule.budget, phi=phi)
    if schedule.family == "pcalm":
        return run_pcalm(
            params,
            scales,
            skips,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=schedule.alpha,
            budget=schedule.budget,
            inner_steps=schedule.inner_steps,
            weight_credit_timing=schedule.weight_credit_timing,
            phi=phi,
        )
    raise ValueError(f"unknown schedule family: {schedule.family}")


def method_grad(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    if schedule.family == "bp":
        return jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
    free, duals = infer_for_schedule(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    return jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)


def _solve_inner(params: Params, scales, skips, x, y, free, duals, state_lr: float, rho: float, steps: int, phi):
    def energy(free_):
        return al_energy_shifted(params, scales, skips, x, y, free_, duals, rho, phi)

    grad_free = jax.grad(energy)

    # `al_energy_shifted` is a *mean over the batch* (the constraint terms divide
    # by batch_size and the supervised loss is a mean), so its gradient w.r.t. a
    # single sample's activity is 1/batch_size of the per-sample gradient. The
    # paper's activity step is the per-sample eta_h = 1/lambda_max; we recover it
    # by scaling the step by the runtime batch size (an effective step of
    # `state_lr * batch_size` on the batch-mean energy). Without this, finite-T PC / PC-ALM
    # inference runs batch_size-times too slowly and never nears its fixed point.
    effective_lr = state_lr * free[0].shape[0]

    def step(free_, _):
        grads = grad_free(free_)
        return [z - effective_lr * g for z, g in zip(free_, grads)], None

    if steps <= 0:
        return free
    free, _ = jax.lax.scan(step, free, xs=None, length=steps)
    return free
