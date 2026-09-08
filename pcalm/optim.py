from __future__ import annotations

import jax
import jax.numpy as jnp


def adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t = jnp.asarray(0, dtype=jnp.int32)
    return m, v, t


def adam_apply(params, grads, state, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: beta2 * vi + (1.0 - beta2) * (gi * gi), v, grads)
    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t
    updates = jax.tree_util.tree_map(lambda mi, vi: (mi / bc1) / (jnp.sqrt(vi / bc2) + eps), m, v)
    params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
    return params, (m, v, t)
