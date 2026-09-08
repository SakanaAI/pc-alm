from __future__ import annotations

import jax
import jax.numpy as jnp

from pcalm.inference import Schedule, constraint_residuals, method_grad, run_pc, run_pcalm
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.training import adam_learning_rate


def small_case():
    params = init_params(jax.random.PRNGKey(0), depth=4, width=5, input_dim=3, output_dim=2)
    scales = model_scales(width=5, depth=4, input_dim=3)
    skips = skip_mask(4)
    phi = activation_fn("tanh")
    x = jax.random.normal(jax.random.PRNGKey(1), (7, 3))
    y = jax.nn.one_hot(jnp.arange(7) % 2, 2)
    return params, scales, skips, phi, x, y


def test_constraints_are_hidden_edges_only():
    params, scales, skips, phi, x, _ = small_case()
    free = [jnp.zeros((x.shape[0], 5)) for _ in range(3)]
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    assert len(residuals) == len(params) - 1
    assert all(r.shape == (x.shape[0], 5) for r in residuals)


def test_pc_has_zero_duals():
    params, scales, skips, phi, x, y = small_case()
    _, duals = run_pc(params, scales, skips, x, y, state_lr=0.1, rho=1.0, steps=2, phi=phi)
    assert all(jnp.allclose(dual, 0.0) for dual in duals)


def test_pcalm_alpha_zero_matches_pc_gradient():
    params, scales, skips, phi, x, y = small_case()
    pc_schedule = Schedule(family="pc", budget=3)
    alm_schedule = Schedule(family="pcalm", budget=3, alpha=0.0)
    g_pc = method_grad(params, scales, skips, x, y, pc_schedule, state_lr=0.1, rho=1.0, phi=phi)
    g_alm = method_grad(params, scales, skips, x, y, alm_schedule, state_lr=0.1, rho=1.0, phi=phi)
    for a, b in zip(g_pc, g_alm):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_pcalm_duals_update_hidden_edges():
    params, scales, skips, phi, x, y = small_case()
    _, duals = run_pcalm(
        params,
        scales,
        skips,
        x,
        y,
        state_lr=0.1,
        rho=1.0,
        alpha=1.0,
        budget=2,
        inner_steps=1,
        weight_credit_timing="post_dual_energy",
        phi=phi,
    )
    assert len(duals) == len(params) - 1
    assert any(float(jnp.linalg.norm(dual)) > 0.0 for dual in duals)


def test_default_adam_lr_uses_width_depth_scaling():
    assert adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=None) == 2e-3
    assert adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=5e-4) == 5e-4


def test_inference_is_per_sample_batch_invariant():
    """Each sample's activity dynamics must be independent of batch size.

    The AL energy is a mean over the batch, so the activity step must scale by
    batch_size to realize the paper's per-sample eta_h. Without that factor a
    sample in a batch of B moves B-times too slowly, so its settled activity
    would depend on batch size. This regression test pins the fix: sample 0's
    settled activity is identical whether solved alone or inside a larger batch.
    """
    params, scales, skips, phi, x, y = small_case()
    kw = dict(state_lr=0.1, rho=1.0, alpha=1.0, budget=3, inner_steps=1,
              weight_credit_timing="pre_dual_energy", phi=phi)
    free_single, _ = run_pcalm(params, scales, skips, x[:1], y[:1], **kw)
    free_batch, _ = run_pcalm(params, scales, skips, x, y, **kw)
    for a, b in zip(free_single, free_batch):
        assert jnp.allclose(a[0], b[0], atol=1e-5, rtol=1e-5)
