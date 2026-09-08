from __future__ import annotations

import jax
import jax.numpy as jnp


def mse_ce_accuracy(logits: jax.Array, y: jax.Array):
    mse = 0.5 * jnp.mean(jnp.sum((logits - y) ** 2, axis=-1))
    ce = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits, axis=-1), axis=-1))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
    return mse, ce, acc


def tree_l2(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(x * x) for x in leaves])))


def tree_dot(a, b) -> jax.Array:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if not leaves_a:
        return jnp.asarray(0.0)
    return jnp.sum(jnp.stack([jnp.sum(x * y) for x, y in zip(leaves_a, leaves_b)]))


def tree_cos(a, b) -> jax.Array:
    return tree_dot(a, b) / jnp.maximum(tree_l2(a) * tree_l2(b), 1e-30)
